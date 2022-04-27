class RunConfig:
    dataset_name = 'stl10'
    arch = 'resnet18'
    epochs = 200
    lr = 0.0003
    learning_rate = lr
    wd = 1e-4
    weight_decay = wd
    seed = None
    disable_cuda = False
    fp16_precision = False
    out_dim = 128
    log_every_n_steps = 100
    temperature = 0.07
    n_views = 2

    data = '/home/zzh/projects/data'
    workers = 12
    batch_size = 256
    gpu_index = 1, 2, 3, 5
    #
    # data = 'd:/projects/data'
    # workers = 0
    # batch_size = 32
    # gpu_index = 0










