from datasets.ucf101 import UCF101



def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):

    training_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'training',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        target_transform=target_transform)
    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    validation_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        'validation',
        opt.n_val_samples,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=opt.sample_duration)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    test_data = UCF101(
        opt.video_path,
        opt.annotation_path,
        subset,
        0,
        spatial_transform,
        temporal_transform,
        target_transform,
        sample_duration=opt.sample_duration)


    return test_data
