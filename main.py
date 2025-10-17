import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datapipe import build_dataset, get_dataset
from model import TriplePath_DANN_Model
import random

def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


subjects = 15
epochs = 1000
classes = 3
Network = TriplePath_DANN_Model
device = torch.device('cuda', 0)
version = 1
confidence_threshold_t1 = 0.7
centroid_weight_threshold_t2 = 0.3
lambda_domain_weight = 0.1
lambda_contrastive_weight = 0.1
temperature_cl = 0.1
lambda_conditional_domain_weight = 0.1
set_random_seed(42)

while True:
    dfile = f'./result/{Network.__name__}_ProtoContrast_LOG_{version:.0f}.csv'
    if not os.path.exists(dfile):
        break
    version += 1

df = pd.DataFrame()
df.to_csv(dfile)

def prototype_contrastive_loss_fn(features, labels, prototypes, temperature, device):
    valid_prototype_mask = torch.norm(prototypes, dim=1) > 1e-8

    if valid_prototype_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    valid_prototypes = prototypes[valid_prototype_mask]
    original_indices_of_valid_prototypes = torch.arange(prototypes.shape[0], device=device)[valid_prototype_mask]

    map_original_idx_to_new_idx = {orig_idx.item(): new_idx for new_idx, orig_idx in
                                   enumerate(original_indices_of_valid_prototypes)}

    sample_filter_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
    mapped_labels_for_loss = []

    for i, label_idx_val in enumerate(labels):
        if label_idx_val.item() in map_original_idx_to_new_idx:
            sample_filter_mask[i] = True
            mapped_labels_for_loss.append(map_original_idx_to_new_idx[label_idx_val.item()])

    if not mapped_labels_for_loss:
        return torch.tensor(0.0, device=device)

    valid_features = features[sample_filter_mask]

    mapped_labels_tensor = torch.tensor(mapped_labels_for_loss, dtype=torch.long, device=device)

    if valid_features.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    sim_matrix = F.cosine_similarity(valid_features.unsqueeze(1), valid_prototypes.unsqueeze(0), dim=2)
    logits = sim_matrix / temperature

    loss_ce = torch.nn.CrossEntropyLoss()
    loss = loss_ce(logits, mapped_labels_tensor)
    return loss


def train(model, train_loader, target_loader, crit, domain_crit, optimizer,
          lambdas_domain_w, lambda_cond_domain_w, lambda_contrastive_w, temp_cl):

    model.train()
    loss_cls_all = 0
    loss_domain_all = 0
    loss_contrastive_all = 0
    total_loss_all = 0
    loss_domain_cond_all = 0
    num_train_batches = 0
    for (source_data, target_data) in zip(train_loader, target_loader):
        source_data = source_data.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()

        source_label_indices = torch.argmax(source_data.y.view(-1, classes), axis=1)
        source_class_out, _, source_global_domain_out, source_features = model(source_data.x, source_data.edge_index,
                                                                               source_data.batch)
        loss_cls = crit(source_class_out, source_label_indices)

        target_class_out, _, target_global_domain_out, target_features = model(target_data.x, target_data.edge_index,
                                                                               target_data.batch)

        source_domain_labels_global = torch.zeros(source_data.num_graphs, dtype=torch.float, device=device)
        target_domain_labels_global = torch.ones(target_data.num_graphs, dtype=torch.float, device=device)
        domain_preds_global = torch.cat([source_global_domain_out, target_global_domain_out])
        domain_labels_global = torch.cat([source_domain_labels_global, target_domain_labels_global])
        loss_domain_global = domain_crit(domain_preds_global, domain_labels_global)

        current_source_prototypes = torch.zeros(classes, source_features.shape[1], device=device)

        for c_idx in range(classes):
            class_mask = (source_label_indices == c_idx)
            if class_mask.sum() > 0:
                current_source_prototypes[c_idx] = source_features[class_mask].mean(dim=0)

        loss_cl_source = prototype_contrastive_loss_fn(source_features, source_label_indices, current_source_prototypes,
                                                       temp_cl, device)

        loss_cl_target = torch.tensor(0.0, device=device)
        reliable_features = torch.tensor([], device=device)
        reliable_labels = torch.tensor([], dtype=torch.long, device=device)
        with torch.no_grad():

            target_probs = F.softmax(target_class_out, dim=1)
            target_max_probs, initial_pseudo_labels = torch.max(target_probs, dim=1)

            confidence_mask = target_max_probs >= confidence_threshold_t1

            if confidence_mask.sum() > 0:
                confident_features = target_features[confidence_mask]
                confident_labels = initial_pseudo_labels[confidence_mask]

                target_prototypes = torch.zeros(classes, target_features.shape[1], device=device)
                for c_idx in range(classes):
                    class_mask = (confident_labels == c_idx)
                    if class_mask.sum() > 0:
                        target_prototypes[c_idx] = confident_features[class_mask].mean(dim=0)

                distances = torch.linalg.norm(confident_features - target_prototypes[confident_labels], dim=1)

                weights = torch.exp(-distances.pow(2))
                proximity_mask = weights >= centroid_weight_threshold_t2

                reliable_features = confident_features[proximity_mask]
                reliable_labels = confident_labels[proximity_mask]

                if reliable_features.shape[0] > 0:
                    loss_cl_target = prototype_contrastive_loss_fn(reliable_features,
                                                                   reliable_labels,
                                                                   current_source_prototypes,
                                                                   temp_cl,
                                                                   device)

        loss_contrastive = (loss_cl_source + loss_cl_target) / 2.0

        loss_domain_conditional = torch.tensor(0.0, device=device)

        source_labels_one_hot = F.one_hot(source_label_indices, num_classes=classes).float()

        if reliable_features.shape[0] > 0:
            target_labels_one_hot = F.one_hot(reliable_labels, num_classes=classes).float()

            alpha = 1.0
            reversed_source_features = model.grl_layer(source_features, alpha)
            reversed_target_features = model.grl_layer(reliable_features, alpha)

            source_domain_pred_cond = model.conditional_domain_classifier(reversed_source_features,
                                                                          source_labels_one_hot)
            target_domain_pred_cond = model.conditional_domain_classifier(reversed_target_features,
                                                                          target_labels_one_hot)

            source_domain_labels_cond = torch.zeros_like(source_domain_pred_cond, device=device)
            target_domain_labels_cond = torch.ones_like(target_domain_pred_cond, device=device)

            domain_preds_cond = torch.cat([source_domain_pred_cond, target_domain_pred_cond])
            domain_labels_cond = torch.cat([source_domain_labels_cond, target_domain_labels_cond])
            loss_domain_conditional = domain_crit(domain_preds_cond, domain_labels_cond)

        total_loss = loss_cls + \
                     lambdas_domain_w * loss_domain_global + \
                     lambda_cond_domain_w * loss_domain_conditional + \
                     lambda_contrastive_w * loss_contrastive

        total_loss.backward()

        optimizer.step()

        loss_cls_all += loss_cls.item() * source_data.num_graphs
        loss_domain_all += loss_domain_global.item() * domain_labels_global.size(0)

        if reliable_features.shape[0] > 0:
            loss_domain_cond_all += loss_domain_conditional.item() * domain_labels_cond.size(0)
        loss_contrastive_all += loss_contrastive.item() * source_data.num_graphs
        total_loss_all += total_loss.item() * source_data.num_graphs
        num_train_batches += source_data.num_graphs

    avg_loss_cls = loss_cls_all / num_train_batches if num_train_batches > 0 else 0
    avg_loss_domain = loss_domain_all / num_train_batches if num_train_batches > 0 else 0
    avg_loss_domain_cond = loss_domain_cond_all / num_train_batches if num_train_batches > 0 else 0
    avg_loss_contrastive = loss_contrastive_all / num_train_batches if num_train_batches > 0 else 0
    avg_total_loss = total_loss_all / num_train_batches if num_train_batches > 0 else 0

    return avg_loss_cls, avg_loss_domain, avg_loss_domain_cond, avg_loss_contrastive, avg_total_loss


def evaluate(model, loader, save_result=False):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            label_one_hot = data.y.view(-1, classes)
            data = data.to(device)
            class_out, pred_probs, _, _ = model(data.x, data.edge_index, data.batch)
            pred_probs_np = pred_probs.detach().cpu().numpy()
            predictions.append(pred_probs_np)
            labels.append(label_one_hot.cpu().numpy())

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    try:
        AUC = roc_auc_score(labels, predictions, average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"Error calculating AUC: {e}. Setting AUC to 0.")
        AUC = 0.0

    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(labels, axis=-1)

    f1 = f1_score(true_classes, predicted_classes, average='macro')
    acc = accuracy_score(true_classes, predicted_classes)
    return AUC, acc, f1


def main():
    build_dataset(subjects)
    result_data = []
    best_acc_results = []

    domain_crit = torch.nn.BCELoss()

    for cv_n in range(subjects):
        best_val_acc = 0.0
        best_epoch = 0

        train_dataset, test_dataset = get_dataset(subjects, cv_n)

        target_domain_dataset = test_dataset

        train_loader = DataLoader(train_dataset, 16, shuffle=True)
        target_loader = DataLoader(target_domain_dataset, 16, shuffle=True)
        test_loader = DataLoader(test_dataset, 16)

        model = TriplePath_DANN_Model(classes=classes).to(device)

        base_lr = 1e-4
        discriminator_lr = 1e-5
        weight_decay_value = 1e-4

        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': base_lr, 'weight_decay': weight_decay_value},
            {'params': model.emotion_classifier.parameters(), 'lr': base_lr, 'weight_decay': weight_decay_value},

            {'params': model.global_domain_classifier.parameters(), 'lr': discriminator_lr,
             'weight_decay': weight_decay_value},
            {'params': model.conditional_domain_classifier.parameters(), 'lr': discriminator_lr,
             'weight_decay': weight_decay_value}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        crit = torch.nn.CrossEntropyLoss()
        patience = 200
        patience_counter = 0

        for epoch in range(epochs):
            avg_loss_cls, avg_loss_domain, avg_loss_domain_cond, avg_loss_contrastive, avg_total_loss = train(
                model, train_loader, target_loader, crit, domain_crit, optimizer,
                lambda_domain_weight, lambda_conditional_domain_weight, lambda_contrastive_weight, temperature_cl
            )
            scheduler.step()
            val_AUC, val_acc, val_f1 = evaluate(model, test_loader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                best_model_path = f'./checkpoints/best_model_cv_{cv_n}.pth'
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
            print(
                f'CV{cv_n:01d}, EP{epoch + 1:03d}, Ls_cls:{avg_loss_cls:.4f}, Ls_dom_g:{avg_loss_domain:.4f}, Ls_dom_c:{avg_loss_domain_cond:.4f}, Ls_cl:{avg_loss_contrastive:.4f} | Val_acc:{val_acc:.4f} | Best: {best_val_acc:.4f}')

            if patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} as validation accuracy did not improve for {patience} epochs.")
                break
        best_acc_results.append(best_val_acc)
        result_data.append([cv_n, best_epoch, best_val_acc])
        current_df = pd.DataFrame(result_data, columns=['Subject', 'Best_Epoch', 'Best_Vacc'])
        current_df.to_csv(dfile, index=False)

    print("\n=== Final Results ===")
    mean_vacc = np.mean(best_acc_results)
    std_vacc = np.std(best_acc_results)
    print(f"Mean Vacc: {mean_vacc:.4f} ± {std_vacc:.4f}")
    print("Individual Results:")
    for subj, acc in enumerate(best_acc_results):
        print(f"Subject {subj:02d}: {acc:.4f}")

    summary_df = pd.DataFrame({
        'Metric': ['Mean Validation Accuracy', 'Std Validation Accuracy'],
        'Value': [mean_vacc, std_vacc]
    })
    summary_df.to_csv(f'./result/{Network.__name__}_ProtoContrast_Summary_{version - 1:.0f}.csv', index=False)


if __name__ == '__main__':
    main()