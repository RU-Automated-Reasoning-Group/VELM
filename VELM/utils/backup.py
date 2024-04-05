    # pdb.set_trace()
    # learn the dso model
    # if args.load_dynamic_model:
    #     with open(f"{args.env}_learned_dynamics.txt", "r") as f:
    #         lines = f.readlines()
    #         learned_dynamic_model = [line[:-1] for line in lines]
    #         print(f"loading dyancmic model from {args.env}_learned_dynamics.txt")
    # else:
    #     # learn a new model
    #     dso = DSO(args)
    #     replay_buffer = neural_agent.replay_buffer
    #     samples = replay_buffer.sample(config_dict["dso_dataset_size"])
    #     X, y_list = process_for_dso(samples, preprocess=args.preprocess)
    #     learned_dynamic_model = dso.train_dso(X, y_list)
    #     learned_dynamic_model = dso.remove_small_numbers()
    #     stds = dso.compute_standard_deviation(X, y_list)
    #     # noise_needed =
    #     #  dso.fit_noise(X, y_list)
    #     # if noise_needed:
    #         # print("noise added to the learned model")
    #     pdb.set_trace()
    #     # learned_dynamic_model = dso.get_learned_model()

    #     # save learned dynamics to be used by the verifier
    #     save_learned_dynamics(args.env, learned_dynamic_model, stds=stds)

    # learned_env = gym.make(pole.env_name)
    # learned_env.load_model()