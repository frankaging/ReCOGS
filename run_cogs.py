from utils.train_utils import *
from datetime import date
import json

if __name__ == '__main__':
    is_notebook = False
    try:
        cmd = argparse.ArgumentParser('The testing components of')
        cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
        cmd.add_argument('--train_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--eval_batch_size', default=128, type=int, help='training batch size')
        cmd.add_argument('--lr', default=0.01, type=float, help='learning rate')
        cmd.add_argument('--data_path', required=True, type=str, help='path to the training corpus')
        cmd.add_argument(
            '--encoder_config_path', 
            type=str, help='path to the encoder config'
        )
        cmd.add_argument(
            '--decoder_config_path', 
            type=str, help='path to the decoder config'
        )
        cmd.add_argument('--max_seq_len', default=512, type=int)
        cmd.add_argument('--seeds', default="42;66;77;88;99", type=str)
        cmd.add_argument('--gradient_accumulation_steps', default=1, type=int)
        cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
        cmd.add_argument('--local_rank', default=-1, type=int, help='multi gpu training')
        cmd.add_argument('--epochs', default=10, type=int, help='training epochs')
        cmd.add_argument('--model_path', type=str, required=False, default=None)
        cmd.add_argument('--model_data_path', type=str, required=False, default="./model/")
        cmd.add_argument('--warm_up', type=float, default=0.1)
        cmd.add_argument('--is_wandb', default=False, action='store_true')
        cmd.add_argument('--spanformer', default=False, action='store_true')
        cmd.add_argument('--log_step', default=10, type=int)
        cmd.add_argument('--valid_steps', default=-1, type=int)
        cmd.add_argument('--early_stopping', default=None, type=int)
        cmd.add_argument('--device', default="cuda", type=str, help='')
        cmd.add_argument('--do_train', default=False, action='store_true')
        cmd.add_argument('--do_eval', default=False, action='store_true')
        cmd.add_argument('--do_test', default=False, action='store_true')
        cmd.add_argument('--do_gen', default=False, action='store_true')
        cmd.add_argument('--least_to_most', default=False, action='store_true')
        cmd.add_argument('--use_glove', default=False, action='store_true')
        cmd.add_argument('--eval_acc', default=False, action='store_true')
        cmd.add_argument('--use_iiem', default=False, action='store_true')
        cmd.add_argument('--output_json', default=False, action='store_true')
        cmd.add_argument('--save_after_epoch', type=int, default=None)
        cmd.add_argument('--lfs', default="cogs", type=str, help='')
        cmd.add_argument('--model_name', default="cogs", type=str, help='')
        
        args = cmd.parse_args(sys.argv[1:])
    except:
        assert False # we only allow this in the DEV mode, not for paper!
        # LSTM settings best: {batch = 512, lr = 8e-4, epoch = 200}
        # Transformer settings best: {batch = 128, lr = 1e-4, epoch = 200}
        is_notebook = True
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        args.gpu = 1
        args.train_batch_size = 512
        args.eval_batch_size = 256
        args.gradient_accumulation_steps = 1
        args.lr = 8e-4
        args.data_path = "./cogs_participle_verb/"
        args.model_data_path = "./model/"
        args.encoder_config_path = None
        args.decoder_config_path = None
        args.max_seq_len = 512
        args.output_dir = "./results_cogs_notebook/"
        args.epochs = 200
        args.warm_up = 0.1
        args.is_wandb = False
        args.log_step = 10
        # args.valid_steps = 500 # -1 not do training eval!
        args.valid_steps = -1
        args.early_stopping = None # large == never early stop!
        args.device = "cuda:0"
        args.spanformer = False
        args.model_path = None
        args.do_train = True
        args.do_eval = False
        args.do_test = True
        args.do_gen = True
        args.least_to_most = False
        args.use_glove = False
        args.eval_acc = False
        args.save_after_epoch = None
        args.use_iiem = False
        args.output_json = False
        args.model_name = "ende_lstm"

        print("Using in a notebook env.")

results = {}

for lf in args.lfs.split(";"):
    for seed in args.seeds.split(";"): # 42, 66, 77, 88, 99
        seed = int(seed)
        set_seed(seed)
        
        data_variant = args.data_path.strip("./")
        args.lf = lf

        model_name = args.model_name
        run_name = f"cogs_pipeline.model.{model_name}.lf.{args.lf}.glove.{args.use_glove}.seed.{seed}"
        if args.do_train == False:
            args.model_path = f"./{args.output_dir}/{run_name}/model-last/"
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        device = torch.device(args.device)
        
        encoder_config_filename = "encoder_config_lstm.json" if model_name == "ende_lstm" else "encoder_config.json"
        decoder_config_filename = "decoder_config_lstm.json" if model_name == "ende_lstm" else "decoder_config.json"
        
        
        if "participle_verb" in args.data_path:
            config_encoder = AutoConfig.from_pretrained(
                os.path.join(args.data_path, encoder_config_filename)
            )
            config_decoder = AutoConfig.from_pretrained(
                    os.path.join(args.data_path, decoder_config_filename) if args.decoder_config_path is None else \
                        args.decoder_config_path
            )
        else:
            config_encoder = AutoConfig.from_pretrained(
                os.path.join(args.model_data_path, encoder_config_filename)
            )
            config_decoder = AutoConfig.from_pretrained(
                    os.path.join(args.model_data_path, decoder_config_filename) if args.decoder_config_path is None else \
                        args.decoder_config_path
            )

        if "participle_verb" in args.data_path:
            src_tokenizer = WordLevelTokenizer(
                os.path.join(args.data_path, "src_vocab.txt"), 
                config_encoder,
                max_seq_len=args.max_seq_len
            )
            tgt_tokenizer = WordLevelTokenizer(
                os.path.join(args.data_path, "tgt_vocab.txt"), 
                config_decoder,
                max_seq_len=args.max_seq_len
            )  
        else:
            src_tokenizer = WordLevelTokenizer(
                os.path.join(args.model_data_path, "src_vocab.txt"), 
                config_encoder,
                max_seq_len=args.max_seq_len
            )
            tgt_tokenizer = WordLevelTokenizer(
                os.path.join(args.model_data_path, "tgt_vocab.txt"), 
                config_decoder,
                max_seq_len=args.max_seq_len
            )

        if args.least_to_most:
            logging.info("Preparing training set to be least to most order.")
        train_dataset = COGSDataset(
            cogs_path=args.data_path, 
            src_tokenizer=src_tokenizer, 
            tgt_tokenizer=tgt_tokenizer, 
            partition=find_partition_name("train", args.lf),
            least_to_most=args.least_to_most
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.train_batch_size, 
            sampler=SequentialSampler(train_dataset),
            collate_fn=train_dataset.collate_batch
        )

        eval_dataset = COGSDataset(
            cogs_path=args.data_path, 
            src_tokenizer=src_tokenizer, 
            tgt_tokenizer=tgt_tokenizer, 
            partition=find_partition_name("dev", args.lf),
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, 
            sampler=SequentialSampler(eval_dataset),
            collate_fn=train_dataset.collate_batch
        )

        test_dataset = COGSDataset(
            cogs_path=args.data_path, 
            src_tokenizer=src_tokenizer, 
            tgt_tokenizer=tgt_tokenizer, 
            partition=find_partition_name("test", args.lf),
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.eval_batch_size, 
            sampler=SequentialSampler(test_dataset),
            collate_fn=train_dataset.collate_batch
        )

        gen_dataset = COGSDataset(
            cogs_path=args.data_path, 
            src_tokenizer=src_tokenizer, 
            tgt_tokenizer=tgt_tokenizer, 
            partition=find_partition_name("gen", args.lf),
        )
        gen_dataloader = DataLoader(
            gen_dataset, batch_size=args.eval_batch_size, 
            sampler=SequentialSampler(gen_dataset),
            collate_fn=train_dataset.collate_batch
        )
        
        if model_name == "ende_transformer":
            logging.info("Baselining the Transformer Encoder-Decoder Model")
            model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            model_config.decoder_start_token_id = config_encoder.bos_token_id
            model_config.pad_token_id = config_encoder.pad_token_id
            model_config.eos_token_id = config_encoder.eos_token_id
            model = EncoderDecoderModel(config=model_config)
        elif model_name == "ende_lstm":
            logging.info("Baselining the LSTM Encoder-Decoder Model")
            model_config = EncoderDecoderConfig.from_encoder_decoder_configs(
                config_encoder, config_decoder
            )
            model_config.decoder_start_token_id = config_encoder.bos_token_id
            model_config.pad_token_id = config_encoder.pad_token_id
            model_config.eos_token_id = config_encoder.eos_token_id
            model = EncoderDecoderLSTMModel(config=model_config)
            
        if args.model_path is not None and model_name == "ende_transformer":
            logging.info("Loading pretrained model.")
            model = model.from_pretrained(args.model_path)
        elif args.model_path is not None and model_name == "ende_lstm":
            logging.info("Loading pretrained model.")
            raw_weights = torch.load(os.path.join(args.model_path, 'pytorch_model.bin'))
            model.load_state_dict(raw_weights)
            
        

        if "cuda:" not in args.device:
            n_gpu = torch.cuda.device_count()
            logging.info(f'__Number CUDA Devices: {n_gpu}')
        else:
            n_gpu = 1
            logging.info(f'__Number CUDA Devices: {n_gpu}')

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        _ = model.to(device)

        t_total = int(len(train_dataloader) * args.epochs)

        warm_up_steps = args.warm_up * t_total
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps,
                                                    num_training_steps=t_total)
        is_master = True
        apex_enable = False                                       
        if not os.path.exists(args.output_dir) and is_master:
            os.mkdir(args.output_dir)

        os.environ["WANDB_PROJECT"] = f"COGS"

        output_dir = os.path.join(args.output_dir, run_name)
        if args.is_wandb:
            import wandb
            run = wandb.init(
                project="COGS-CKY-Transformer", 
                entity="wuzhengx",
                name=run_name,
            )
            wandb.config.update(args)
        if not os.path.exists(args.output_dir) and is_master:
            os.mkdir(args.output_dir)

        trainer = COGSTrainer(
            model, device=device, 
            src_tokenizer=src_tokenizer, 
            tgt_tokenizer=tgt_tokenizer, 
            logger=logger,
            is_master=is_master, 
            n_gpu=n_gpu,
            is_wandb=args.is_wandb, 
            model_name=model_name,
            eval_acc=args.eval_acc,
        )
        num_params = count_parameters(model)
        logging.info(f'Number of model params: {num_params}')

        if args.do_train:
            logging.info(f"OUTPUT DIR: {output_dir}")
            trainer.train(
                train_dataloader, eval_dataloader,
                optimizer, scheduler, 
                log_step=args.log_step, valid_steps=args.valid_steps,
                output_dir=output_dir, epochs=args.epochs, 
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                save_after_epoch=args.save_after_epoch,
            )
        
        if args.do_test and not args.use_iiem:
            trainer.model.eval()
            epoch_iterator = tqdm(test_dataloader, desc="Iteration", position=0, leave=True)
            total_count = 0
            correct_count = 0
            for step, inputs in enumerate(epoch_iterator):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = inputs["labels"].to(device)
                if model_name == "ende_lstm":
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=model_config.eos_token_id,
                        max_length=args.max_seq_len,
                    )
                decoded_preds = tgt_tokenizer.batch_decode(outputs)
                decoded_labels = tgt_tokenizer.batch_decode(labels)

                for i in range(len(decoded_preds)):
                    if decoded_preds[i] == decoded_labels[i]:
                        correct_count += 1
                    else:
                        pass
                        # print(decoded_preds[i])
                        # print(decoded_labels[i])
                    total_count += 1
                current_acc = round(correct_count/total_count, 2)
                epoch_iterator.set_postfix({'acc': current_acc})
            test_acc = current_acc

        if args.do_gen and not args.use_iiem:
            per_cat_eval = {}
            for cat in set(gen_dataset.eval_cat):
                per_cat_eval[cat] = [0, 0] # correct, total
            trainer.model.eval()
            epoch_iterator = tqdm(gen_dataloader, desc="Iteration", position=0, leave=True)
            total_count = 0
            correct_count = 0
            for step, inputs in enumerate(epoch_iterator):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = inputs["labels"].to(device)
                if model_name == "ende_lstm":
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=model_config.eos_token_id,
                        max_length=args.max_seq_len,
                    )
                decoded_preds = tgt_tokenizer.batch_decode(outputs)
                decoded_labels = tgt_tokenizer.batch_decode(labels)

                input_labels = src_tokenizer.batch_decode(input_ids)
                for i in range(len(decoded_preds)):
                    cat = gen_dataset.eval_cat[total_count]
                    if decoded_preds[i] == decoded_labels[i]:
                        correct_count += 1
                        per_cat_eval[cat][0] += 1
                    else:
                        if cat == "prim_to_obj_proper":
                            pass
                            # print("input: ", input_labels[i])
                            # print("pred: ", decoded_preds[i])
                            # print("actual: ", decoded_labels[i])
                            # print("cat: ", cat)
                            # print()
                    total_count += 1
                    per_cat_eval[cat][1] += 1
                current_acc = correct_count/total_count
                epoch_iterator.set_postfix({'acc': current_acc})

            struct_pp_acc = 0
            struct_cp_acc = 0
            struct_obj_subj_acc = 0

            lex_acc = 0
            lex_count = 0
            for k, v in per_cat_eval.items():
                if k  == "pp_recursion":
                    struct_pp_acc = 100 * v[0]/v[1]
                elif k  == "cp_recursion":
                    struct_cp_acc = 100 * v[0]/v[1]
                elif k  == "obj_pp_to_subj_pp":
                    struct_obj_subj_acc = 100 * v[0]/v[1]
                elif k  == "subj_to_obj_proper":
                    subj_to_obj_proper_acc = 100 * v[0]/v[1]
                elif k  == "prim_to_obj_proper":
                    prim_to_obj_proper_acc = 100 * v[0]/v[1]
                elif k  == "prim_to_subj_proper": 
                    prim_to_subj_proper_acc = 100 * v[0]/v[1]
                else:
                    lex_acc += v[0]
                    lex_count += v[1]
            lex_acc /= lex_count
            lex_acc *= 100
            current_acc *= 100

            print(f"obj_pp_to_subj_pp: {struct_obj_subj_acc}")
            print(f"cp_recursion: {struct_cp_acc}")
            print(f"pp_recursion: {struct_pp_acc}")
            print(f"subj_to_obj_proper: {subj_to_obj_proper_acc}")
            print(f"prim_to_obj_proper: {prim_to_obj_proper_acc}")
            print(f"prim_to_subj_proper: {prim_to_subj_proper_acc}")
            print(f"LEX: {lex_acc}")
            print(f"OVERALL: {current_acc}")

            results[f"{seed}_{data_variant}_{lf}"] = {
                "obj_pp_to_subj_pp" : struct_obj_subj_acc,
                "cp_recursion" : struct_cp_acc,
                "pp_recursion" : struct_pp_acc,
                "subj_to_obj_proper" : subj_to_obj_proper_acc,
                "prim_to_obj_proper" : prim_to_obj_proper_acc,
                "prim_to_subj_proper" : prim_to_subj_proper_acc,
                "lex_acc" : lex_acc,
                "overall_acc" : current_acc,
                "test_acc" : test_acc
            }

        if args.do_test and args.use_iiem:
            trainer.model.eval()
            epoch_iterator = tqdm(test_dataloader, desc="Iteration", position=0, leave=True)
            total_count = 0
            correct_count = 0
            for step, inputs in enumerate(epoch_iterator):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = inputs["labels"].to(device)
                if model_name == "ende_lstm":
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=model_config.eos_token_id,
                        max_length=args.max_seq_len,
                    )
                decoded_preds = tgt_tokenizer.batch_decode(outputs)
                decoded_labels = tgt_tokenizer.batch_decode(labels)

                for i in range(len(decoded_preds)):

                    # use conj-based eval function.
                    if check_equal(decoded_labels[i], decoded_preds[i]):
                        correct_count += 1
                    else:
                        pass
                        # print(decoded_preds[i])
                        # print(decoded_labels[i])

                    total_count += 1
                current_acc = round(correct_count/total_count, 2)
                epoch_iterator.set_postfix({'acc': current_acc})
            test_acc = current_acc
            
        if args.do_gen and args.use_iiem:
            per_cat_eval = {}
            for cat in set(gen_dataset.eval_cat):
                per_cat_eval[cat] = [0, 0] # correct, total
            trainer.model.eval()
            epoch_iterator = tqdm(gen_dataloader, desc="Iteration", position=0, leave=True)
            total_count = 0
            correct_count = 0
            for step, inputs in enumerate(epoch_iterator):
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = inputs["labels"].to(device)
                if model_name == "ende_lstm":
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                    )
                else:
                    outputs = trainer.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        eos_token_id=model_config.eos_token_id,
                        max_length=args.max_seq_len,
                    )
                decoded_preds = tgt_tokenizer.batch_decode(outputs)
                decoded_labels = tgt_tokenizer.batch_decode(labels)

                input_labels = src_tokenizer.batch_decode(input_ids)
                for i in range(len(decoded_preds)):

                    # use conj-based eval function.
                    cat = gen_dataset.eval_cat[total_count]
                    if check_equal(decoded_labels[i], decoded_preds[i]):
                        correct_count += 1
                        per_cat_eval[cat][0] += 1
                        if cat == "obj_pp_to_subj_pp":
                            pass
                    else:
                        if cat == "obj_pp_to_subj_pp":
                            pass
                            # print("input: ", input_labels[i])
                            # print("pred: ", decoded_preds_ii_str)
                            # print("actual: ", decoded_labels_ii_str)
                            # print("cat: ", cat)
                            # print()
                    total_count += 1
                    per_cat_eval[cat][1] += 1
                current_acc = correct_count/total_count
                epoch_iterator.set_postfix({'acc': current_acc})

            struct_pp_acc = 0
            struct_cp_acc = 0
            struct_obj_subj_acc = 0

            lex_acc = 0
            lex_count = 0
            for k, v in per_cat_eval.items():
                if k  == "pp_recursion":
                    struct_pp_acc = 100 * v[0]/v[1]
                elif k  == "cp_recursion":
                    struct_cp_acc = 100 * v[0]/v[1]
                elif k  == "obj_pp_to_subj_pp":
                    struct_obj_subj_acc = 100 * v[0]/v[1]
                elif k  == "subj_to_obj_proper":
                    subj_to_obj_proper_acc = 100 * v[0]/v[1]
                elif k  == "prim_to_obj_proper":
                    prim_to_obj_proper_acc = 100 * v[0]/v[1]
                elif k  == "prim_to_subj_proper":
                    prim_to_subj_proper_acc = 100 * v[0]/v[1]
                else:
                    lex_acc += v[0]
                    lex_count += v[1]
            lex_acc /= lex_count
            lex_acc *= 100
            current_acc *= 100

            print(f"obj_pp_to_subj_pp: {struct_obj_subj_acc}")
            print(f"cp_recursion: {struct_cp_acc}")
            print(f"pp_recursion: {struct_pp_acc}")
            print(f"subj_to_obj_proper: {subj_to_obj_proper_acc}")
            print(f"prim_to_obj_proper: {prim_to_obj_proper_acc}")
            print(f"prim_to_subj_proper: {prim_to_subj_proper_acc}")
            print(f"LEX: {lex_acc}")
            print(f"OVERALL: {current_acc}")

            results[f"{seed}_{data_variant}_{lf}"] = {
                "obj_pp_to_subj_pp" : struct_obj_subj_acc,
                "cp_recursion" : struct_cp_acc,
                "pp_recursion" : struct_pp_acc,
                "subj_to_obj_proper" : subj_to_obj_proper_acc,
                "prim_to_obj_proper" : prim_to_obj_proper_acc,
                "prim_to_subj_proper" : prim_to_subj_proper_acc,
                "lex_acc" : lex_acc,
                "overall_acc" : current_acc,
                "test_acc" : test_acc
            }

    if args.output_json:
        today = date.today()
        date = today.strftime("%b-%d-%Y")
        json_object = json.dumps(results, indent=4)
        data_filename = date+"_"+model_name+"_"+data_variant+"_"+lf
        with open(os.path.join(args.output_dir, f"{data_filename}.json"), "w") as outfile:
            outfile.write(json_object)

