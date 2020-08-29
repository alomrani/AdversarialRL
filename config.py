
class config:
    sampling_ratio = 30
    # -------------learning_related--------------------#
    batch_size = 15
    workers = 0
    iter_size = 1
    num_episodes = 100000
    test_episodes = 500
    save_episodes = 500
    resume_model = 'model/8_29_14_15000.pth'  # 'model/8_28_21_16000.pth'
    display = 2
    # -------------rl_related--------------------#
    pi_loss_coeff = 1.0
    v_loss_coeff = 1.0
    beta = 0
    c_loss_coeff = 0.5  # 0.005
    switch = 4
    warm_up_episodes = 100000
    episode_len = 20
    gamma = 1.0
    reward_method = 'abs'
    noise_scale = 0.2  # 0.5
    # -------------continuous parameters--------------------#
    actions = {
        "addition": 1,
        'subtraction': 2,
    }
    num_actions = len(actions) + 1

    # -------------lr_policy--------------------#
    base_lr = 0.00001
    # poly
    lr_policy = 'exp'
    policy_parameter = {
      'gamma': 0.99,
      'max_iter': 40000,
    }

    # -------------folder--------------------#
    dataset = 'MICCAI'
    root = 'MICCAI/data/'
