
class config:
    sampling_ratio = 30
    # -------------learning_related--------------------#
    batch_size = 1
    workers = 4
    iter_size = 2
    num_episodes = 10000
    test_episodes = 500
    save_episodes = 10000
    resume_model = ''  # 'model/8_28_21_16000.pth'
    display = 10
    # -------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 0.25
    beta = 0.1
    c_loss_coeff = 0.5  # 0.005
    switch = 4
    warm_up_episodes = 1000
    episode_len = 50
    gamma = 1
    reward_method = 'abs'
    noise_scale = 0.2  # 0.5
    # -------------continuous parameters--------------------#
    actions = {
        "addition": 1,
        'subtraction': 2,
    }
    num_actions = len(actions) + 1

    # -------------lr_policy--------------------#
    base_lr = 0.001
    # poly
    lr_policy = 'poly'
    policy_parameter = {
      'power': 1,
      'max_iter': 40000,
    }

    # -------------folder--------------------#
    dataset = 'MICCAI'
    root = 'MICCAI/data/'
