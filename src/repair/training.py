""" Utilities for training datasets. """

def _train(self, dataset_dict):
    trainer = self._get_trainer(dataset_dict)
    args = trainer.args
    
    if self.config.training.wandb_hp_space: # Sweep for good hyperparameters
        print("Looking for best hyperparameters")
        best_trial = self._search_best_training_params(trainer)
        training_params = best_trial.hyperparameters
    elif self.config.training.hyperparameters: # Use the specified hyperaparameters
        training_params = self.config.training.hyperparameters
    else: # use the default trainer hyperparameters
        # TODO: does not work apparently? fix that 
        training_params = args.to_dict()
    
    # Update the training arguments with the set hyperparameters 
    training_params = {k: v for k, v in training_params.items() if hasattr(args, k)}
    print("Best training parameters", training_params)

    trainer = self._get_trainer(dataset_dict, **training_params)
    print("Retraining the model for a final time.")
    trainer.train()
    trainer.save_model(self.model_save_dir)
    print("Saving model to", self.model_save_dir)

    return trainer.model 


def _search_best_training_params(self, trainer):
    n_trials = 2 if self.test_run else self.config.training.n_trials 
    best_trial = trainer.hyperparameter_search(
            backend="wandb",
            hp_space=lambda trial: self.config.training.wandb_hp_space.toDict(),
            n_trials=n_trials,
    )

    return best_trial

def load_trained_model(self):
    device = self.device #self.accelerator.device
    return self.agent.load_model(self.model_save_dir).to(device)
