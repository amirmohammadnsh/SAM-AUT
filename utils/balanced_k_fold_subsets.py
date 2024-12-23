import numpy as np
import json
import os
from collections import defaultdict

from dataset import PAUTFLAW2

from sklearn.model_selection import BaseCrossValidator,train_test_split
from sklearn.utils import check_random_state
from torch.utils.data import Subset,Dataset

class AnnotationBalancedKFoldPAUTFLAW:
    def __init__(self, dataset_root,split,filter_empty,preprocess,task_name, n_splits=5, shuffle=True, test_size=0.2,use_bins=False, random_state=42,loading=False,train_data_portion=1.0):
        self.dataset_root = dataset_root
        self.split = split
        self.filter_empty = filter_empty
        self.preprocess = preprocess
        self.dataset = PAUTFLAW2(self.dataset_root,self.split,self.filter_empty,self.preprocess)
        self.task_name= task_name
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.base_dir="./k_fold_dataset"
        self.fold_results = defaultdict(list)
        self.test_size = test_size
        self.use_bins = use_bins
        self.train_data_portion = train_data_portion
        self.train_data_portion_images = 0
        self.train_data_portion_annts = 0

        # Count annotations per image
        self.annotation_counts = np.array([len(self.dataset.coco.getAnnIds(imgIds=img_id)) for img_id in self.dataset.ids])
        self.train_val_indices, self.test_indices = self._train_test_split()
        # Create AnnotationBalancedKFold object
        self.abkf = AnnotationBalancedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        # Generate and store the splits
        # self.folds = list(self.abkf.split(np.arange(len(self.dataset)), self.annotation_counts))
        self.folds = list(self.abkf.split(self.train_val_indices, self.annotation_counts[self.train_val_indices]))

        if not loading:
            self.save_folds()
        if self.train_data_portion == 1.0:
            self.print_split_fold_stats()

    def _train_test_split(self):
        indices = np.arange(len(self.dataset))
        if self.use_bins:
            bins = [1,3,6]  # Adjust these bins as needed
            binned_counts = np.digitize(self.annotation_counts, bins)
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.annotation_counts if not self.use_bins else binned_counts,
            shuffle=self.shuffle
        )        
        
        return train_val_indices, test_indices

    def get_test_set(self):
        return CustomSubset(self.dataset, self.test_indices)



    def get_fold(self, fold_idx):
        if fold_idx < 0 or fold_idx >= self.n_splits:
            raise ValueError(f"fold_idx should be between 0 and {self.n_splits - 1}")
        
        train_indices, val_indices = self.folds[fold_idx]
        if self.train_data_portion < 1.0:
            # Calculate the number of training samples to use
            train_annotation_counts = self.annotation_counts[self.train_val_indices[train_indices]]
            total_annotations = train_annotation_counts.sum()
            target_annotations = int(total_annotations * self.train_data_portion)
            
            # Select indices until we reach the target number of annotations
            cumulative_annotations = 0
            selected_train_indices = []
            for idx in train_indices:
                selected_train_indices.append(idx)
                cumulative_annotations += self.annotation_counts[self.train_val_indices[idx]]
                if cumulative_annotations >= target_annotations:
                    self.train_data_portion_images = len(selected_train_indices)
                    self.train_data_portion_annts = cumulative_annotations
                    break
        else:
            selected_train_indices = train_indices
            
        train_subset = CustomSubset(self.dataset, self.train_val_indices[selected_train_indices])

        # train_subset = CustomSubset(self.dataset, self.train_val_indices[train_indices])
        val_subset = CustomSubset(self.dataset, self.train_val_indices[val_indices])
        
        if self.train_data_portion < 1.0:
            print(f"Fold {fold_idx + 1}:")
            
            print(f"Used train set: {self.train_data_portion_images} images, {self.train_data_portion_annts} annotations")

            val_annotations = self.annotation_counts[self.train_val_indices[val_indices]].sum()
            print(f"Validation set: {len(val_indices)} images, {val_annotations} annotations")

            self.fold_results[fold_idx] = {
                'f1_score': 0,
                'num_samples': val_annotations
            }
            print(f"Annotations ratio (val/train): {val_annotations/self.train_data_portion_annts:.2f}")
            sample_val_ids = [self.dataset.ids[idx] for idx in val_indices[:5]]
            print(f"Sample image IDs in validation set: {sample_val_ids}")

            test_annotations = self.annotation_counts[self.test_indices].sum()
            print(f"Test set: {len(self.test_indices)} images, {test_annotations} annotations")

            print(f"Annotations ratio (test/train): {test_annotations/(val_annotations+self.train_data_portion_annts):.2f}")
            print()
            return train_subset, val_subset
        else:
            return train_subset, val_subset

    def print_split_fold_stats(self):
        train_val_annotations = self.annotation_counts[self.train_val_indices].sum()
        test_annotations = self.annotation_counts[self.test_indices].sum()
        print("Initial Train-Test Split:")
        print(f"  Train set: {len(self.train_val_indices)} images, {train_val_annotations} annotations")
        print(f"  Test set: {len(self.test_indices)} images, {test_annotations} annotations")
        print(f"  Annotations ratio (test/train): {test_annotations/train_val_annotations:.2f}")
        print("\nCross-Validation Folds:")                
        for i, (train_indices, val_indices) in enumerate(self.folds):
            train_annotations = self.annotation_counts[self.train_val_indices[train_indices]].sum()
            val_annotations = self.annotation_counts[self.train_val_indices[val_indices]].sum()
            print(f"Fold {i + 1}:")
            print(f"  Train set: {len(train_indices)} images, {train_annotations} annotations")
            print(f"  Validation set: {len(val_indices)} images, {val_annotations} annotations")
            self.fold_results[i] = {
                'f1_score': 0,
                'num_samples': val_annotations
            }
            print(f"  Annotations ratio (val/train): {val_annotations/train_annotations:.2f}")
            sample_val_ids = [self.dataset.ids[idx] for idx in val_indices[:5]]
            print(f"  Sample image IDs in validation set: {sample_val_ids}")
            print()

    def save_folds(self):
        """Save the fold indices to a JSON file."""
        save_path = os.path.join(self.base_dir, self.task_name)
        os.makedirs(save_path, exist_ok=True)
        
        folds_data = {
            "dataset_root": self.dataset_root,
            "split":self.split,
            "filter_empty":self.filter_empty,
            "preprocess":self.preprocess,
            "task_name": self.task_name,
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "use_bins":self.use_bins,
            "test_size":self.test_size,
            "train_val_indices":self.train_val_indices.tolist(),
            "test_indices":self.test_indices.tolist(),
            "folds": [{"train": train.tolist(), "val": val.tolist()} for train, val in self.folds]
        }
        
        with open(os.path.join(save_path, "folds_dataset.json"), 'w') as f:
            json.dump(folds_data, f)
        
        print(f"Folds saved to {save_path}")

    @classmethod
    def load_folds(cls, load_path):
        """Load the fold indices from a JSON file and return a new instance."""
        with open(load_path, 'r') as f:
            folds_data = json.load(f)
        
        instance = cls(
            dataset_root= folds_data["dataset_root"],
            split=folds_data["split"],
            filter_empty = folds_data["filter_empty"],
            preprocess = folds_data["preprocess"],
            task_name = folds_data["task_name"],
            n_splits=folds_data["n_splits"],
            test_size = folds_data["test_size"],
            use_bins = folds_data["use_bins"],
            shuffle=folds_data["shuffle"],
            random_state=folds_data["random_state"],
            loading=True
        )
        instance.train_val_indices = np.array(folds_data["train_val_indices"])
        instance.test_indices = np.array(folds_data["test_indices"])        
        instance.folds = [(np.array(fold["train"]), np.array(fold["val"])) for fold in folds_data["folds"]]
        
        print(f"Folds loaded from {load_path}")
        return instance

    
class AnnotationBalancedKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        annotation_counts = np.array(y)
        fold_annotation_counts = np.zeros(self.n_splits)
        folds = [[] for _ in range(self.n_splits)]

        # Sort indices by annotation count (descending)
        sorted_indices = indices[np.argsort(-annotation_counts)]

        for idx in sorted_indices:
            # Find fold with least annotations
            target_fold = np.argmin(fold_annotation_counts)
            folds[target_fold].append(idx)
            fold_annotation_counts[target_fold] += annotation_counts[idx]

 
        for fold in folds:
            yield np.array(fold)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        for test_index in self._iter_test_indices(X, y, groups):
            train_index = indices[~np.isin(indices, test_index)]
            yield train_index, test_index

class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)