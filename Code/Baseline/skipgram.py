import argsparser, dataloaders, evaluation, train, models
import torch
import os

# Parse the arguments
args = argsparser.parse_args()

def skipgram():
    print("!!========================CREATING DATA LOADERS========================!!")
    dataloader, _ , _ , vocab_size = dataloaders \
    .create_data_loaders(args, embedding_type = "NAE")
    print("!!========================INSTANTIATING MODEL========================!!")
    # Instantiate your model
    model = models.SkipGram(vocab_size = vocab_size, embedding_dim = args.embedding_dim)

    # Define your loss function and optimizer
    print("!!========================TRAINING MODEL========================!!")
    trained_model = train.train_skipGram (dataloader, model, vocab_size, args)
    torch.save(trained_model.state_dict(), \
               os.path.join(args.skipgram_file_path, \
                            f"skipgram_weights{args.prompt}_{args.embedding_dim}.pth"))
    
if __name__ == "__main__":
    skipgram()