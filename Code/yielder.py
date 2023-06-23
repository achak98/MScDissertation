import torch

def yield_hyps(args,embedding_types):
    for embedding_type in embedding_types:
        for stride1 in range(args.cnn_window_size_small, 1, -1):
            for ks1 in range(args.cnn_window_size_small, 1, -1):
                for stride2 in range (args.cnn_window_size_small, 1, -1):
                    for stride3 in range (args.cnn_window_size_medium, 1, -1):
                        for ks2 in range (args.cnn_window_size_medium, ks1, -1):
                            for stride4 in range (args.cnn_window_size_medium, ks1, -1):
                                for stride5 in range (args.cnn_window_size_large, 1, -1):
                                    for ks3 in range (args.cnn_window_size_large, ks2, -1):
                                        for stride6 in range (args.cnn_window_size_large, ks2, -1):
                                            strides = [stride1, stride2, stride3, stride4, stride5, stride6]
                                            kernels = [ks1, ks2, ks3]
                                            yield (embedding_type, strides, kernels)

def yield_crossval_dls(args, dataset=None,k_fold=10):
    
    total_size = len(dataset)
    fraction = 1/k_fold
    seg = int(total_size * fraction)
   
    test_size = int(0.2 * total_size)
    val_size = int(0.1 * total_size)
    for i in range(k_fold):
        trll = 0
        trlr = i * seg
        testl = trlr
        testr = trlr + test_size
        trrl = testr
        trrr = total_size
        
        train_left_indices = list(range(trll,trlr))
        train_right_indices = list(range(trrl,trrr))
        
        train_indices = train_left_indices + train_right_indices
        test_indices = list(range(testl,testr))
        val_indices = train_indices[-val_size:]
        train_indices = train_indices[:val_size]
        
        train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
        val_set = torch.utils.data.dataset.Subset(dataset,val_indices)
        test_set = torch.utils.data.dataset.Subset(dataset,test_indices)
        print(f"num of workers : {args.numOfWorkers} and type: {type(args.numOfWorkers)}")
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.numOfWorkers)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.numOfWorkers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.numOfWorkers)
        dataloaders = [train_loader, val_loader, test_loader]
        yield dataloaders

