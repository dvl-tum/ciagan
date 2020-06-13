from train import ciagan_exp

r = ciagan_exp.run(config_updates={
    'TRAIN_PARAMS':
        {
            'ARCH_NUM': 'unet_flex',
            'ARCH_SIAM': 'resnet_siam',
            'EPOCH_START': 0, 'EPOCHS_NUM': 2001,
            'LEARNING_RATE': 0.0001,
            'FILTER_NUM': 16,

            'ITER_CRITIC': 1,
            'ITER_GENERATOR': 3,
            'ITER_SIAMESE': 1,

            'GAN_TYPE': 'lsgan',  # lsgan wgangp
        },
    'DATA_PARAMS':
        {
            'LABEL_NUM': 5,
            'BATCH_SIZE': 1,
            'WORKERS_NUM': 4,
            'IMG_SIZE': 64,
        },
    'OUTPUT_PARAMS': {
            'SAVE_EPOCH': 1,
            'SAVE_CHECKPOINT': 100,
            'LOG_ITER': 2,
            'COMMENT': "Something here",
            'EXP_TRY': 'check',
        }
    })