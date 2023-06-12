import argsparser, dataloaders, evaluation, train, models


# Parse the arguments
args = argsparser.parse_args()

def baseline():
    print("!!========================CREATING DATA LOADERS========================!!")
    train_dataloader, val_dataloader, test_dataloader, _ = dataloaders \
    .create_data_loaders(args, embedding_type = "skipgram")
    
    #print("VOCAB_SIZE: ", vocab_size)
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.Baseline(args)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    model = train.train(model, train_dataloader, val_dataloader, args.num_epochs, args.lr)
    print("!!========================EVALUATING MODEL========================!!")
    qwk_score = evaluation.evaluate(model, test_dataloader)
    print("Quadratic Weighted Kappa (QWK) Score on test set:", qwk_score)

if __name__ == "__main__":
    baseline()