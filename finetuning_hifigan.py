# import argparse
# import json
# import logging
# from pathlib import Path
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.multiprocessing as mp
# from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
# import soundfile as sf
# from tqdm import tqdm
# import pandas as pd

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths to the input files and directories
# train_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_train.tsv'
# valid_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_valid.tsv'
# train_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_train.txt'
# valid_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_valid.txt'
# vocoder_checkpoint = '/home/taruntejaneurips23/s2s_mtp/vocoder/g_00500000'
# vocoder_config = '/home/taruntejaneurips23/s2s_mtp/vocoder/config.json'
# checkpoint_dir = Path('/home/taruntejaneurips23/DEMO/hifi_gan/hifigan/hifigan')
# results_path = Path('/home/taruntejaneurips23/DEMO/evalution_hifigan_audio')

# class UnitSequenceDataset(Dataset):
#     def __init__(self, units_txt, tsv_file):
#         self.unit_sequences = self.load_unit_sequences(units_txt)
#         self.audio_paths = self.load_audio_paths(tsv_file)

#     def load_unit_sequences(self, filepath):
#         with open(filepath, 'r') as f:
#             unit_sequences = [list(map(int, line.strip().split())) for line in f]
#         return unit_sequences

#     def load_audio_paths(self, filepath):
#         df = pd.read_csv(filepath, sep='\t')
#         return df['tgt_audio'].tolist()

#     def __len__(self):
#         return len(self.unit_sequences)

#     def __getitem__(self, idx):
#         audio, sr = sf.read(self.audio_paths[idx])
#         return {
#             'unit_sequence': torch.LongTensor(self.unit_sequences[idx]),
#             'audio': torch.FloatTensor(audio)
#         }

# def dump_result(output_path, sample_id, pred_wav, sample_rate=16000):
#     sf.write(f"{output_path}/{sample_id}_pred.wav", pred_wav.detach().cpu().numpy(), sample_rate)

# def collate_fn(batch):
#     unit_sequences = [item['unit_sequence'] for item in batch]
#     audio_waveforms = [item['audio'] for item in batch]

#     max_length = max([len(audio) for audio in audio_waveforms])
#     padded_waveforms = [F.pad(audio, (0, max_length - len(audio))) for audio in audio_waveforms]

#     unit_sequences_padded = torch.nn.utils.rnn.pad_sequence(unit_sequences, batch_first=True, padding_value=0)
#     return {
#         'unit_sequences': unit_sequences_padded,
#         'audio_waveforms': torch.stack(padded_waveforms)
#     }

# def train(rank, world_size, args):
#     dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="tcp://localhost:54321")

#     log_dir = args.checkpoint_dir / "logs"
#     log_dir.mkdir(exist_ok=True, parents=True)
#     writer = SummaryWriter(log_dir) if rank == 0 else None

#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda(rank)
#     vocoder = DDP(vocoder, device_ids=[rank])

#     optimizer = optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

#     train_dataset = UnitSequenceDataset(args.train_units_txt, args.train_tsv)
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=collate_fn)

#     valid_dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     best_loss = float("inf")
#     global_step = 0

#     for epoch in range(args.epochs):
#         train_sampler.set_epoch(epoch)
#         vocoder.train()
#         for i, batch in enumerate(train_dataloader):
#             unit_seq = batch['unit_sequences'].to(rank)
#             target_audio = batch['audio_waveforms'].to(rank)
#             optimizer.zero_grad()
#             generated_wav = vocoder({"code": unit_seq})["wav"]
#             loss = F.mse_loss(generated_wav, target_audio)
#             loss.backward()
#             optimizer.step()

#             global_step += 1
#             if rank == 0 and global_step % args.log_interval == 0:
#                 writer.add_scalar("train/loss", loss.item(), global_step)
#                 logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")

#         scheduler.step()

#         # Validation step
#         vocoder.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for i, batch in enumerate(valid_dataloader):
#                 unit_seq = batch['unit_sequences'].to(rank)
#                 target_audio = batch['audio_waveforms'].to(rank)
#                 generated_wav = vocoder({"code": unit_seq})["wav"]
#                 loss = F.mse_loss(generated_wav, target_audio)
#                 val_loss += loss.item()
#             val_loss /= len(valid_dataloader)

#         if rank == 0:
#             writer.add_scalar("val/loss", val_loss, epoch)
#             logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")

#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 save_path = args.checkpoint_dir / f"best_checkpoint.pt"
#                 torch.save(vocoder.state_dict(), save_path)
#                 logger.info(f"New best checkpoint saved at epoch {epoch} with loss {val_loss}")

#     dist.destroy_process_group()

# def evaluate(args):
#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     results_path = Path(args.results_path)
#     results_path.mkdir(parents=True, exist_ok=True)

#     vocoder.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(dataloader)):
#             unit_seq = batch['unit_sequences'].cuda()
#             pred_wav = vocoder({"code": unit_seq})["wav"]
#             sample_id = Path(batch['audio_waveforms'][0]).stem
#             dump_result(results_path, sample_id, pred_wav)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train-tsv", type=str, default=train_tsv, help="Path to the train TSV file")
#     parser.add_argument("--valid-tsv", type=str, default=valid_tsv, help="Path to the validation TSV file")
#     parser.add_argument("--train-units-txt", type=str, default=train_units_txt, help="Path to the train units text file")
#     parser.add_argument("--valid-units-txt", type=str, default=valid_units_txt, help="Path to the validation units text file")
#     parser.add_argument("--vocoder-checkpoint", type=str, default=vocoder_checkpoint, help="Path to the vocoder checkpoint")
#     parser.add_argument("--vocoder-config", type=str, default=vocoder_config, help="Path to the vocoder config file")
#     parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, help="Directory to save checkpoints")
#     parser.add_argument("--results-path", type=str, default=results_path, help="Directory to save results")
#     parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for fine-tuning")
#     parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=3100, help="Number of epochs for training")
#     parser.add_argument("--log-interval", type=int, default=5, help="Interval for logging training metrics")
#     parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Interval for saving checkpoints")
#     parser.add_argument("--eval", action="store_true", help="Evaluate the model after training")

#     args = parser.parse_args()

#     if not args.eval:
#         world_size = torch.cuda.device_count()
#         mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
#     else:
#         evaluate(args)



# import argparse
# import json
# import logging
# from pathlib import Path
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
# import soundfile as sf
# from tqdm import tqdm
# import pandas as pd

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths to the input files and directories
# train_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_train.tsv'
# valid_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_valid.tsv'
# train_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_train.txt'
# valid_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_valid.txt'
# vocoder_checkpoint = '/home/taruntejaneurips23/s2s_mtp/vocoder/g_00500000'
# vocoder_config = '/home/taruntejaneurips23/s2s_mtp/vocoder/config.json'
# checkpoint_dir = Path('/home/taruntejaneurips23/DEMO/hifi_gan/hifigan/hifigan')
# results_path = Path('/home/taruntejaneurips23/DEMO/evalution_hifigan_audio')

# class UnitSequenceDataset(Dataset):
#     def __init__(self, units_txt, tsv_file):
#         self.unit_sequences = self.load_unit_sequences(units_txt)
#         self.audio_paths = self.load_audio_paths(tsv_file)

#     def load_unit_sequences(self, filepath):
#         with open(filepath, 'r') as f:
#             unit_sequences = [list(map(int, line.strip().split())) for line in f]
#         return unit_sequences

#     def load_audio_paths(self, filepath):
#         df = pd.read_csv(filepath, sep='\t')
#         return df['tgt_audio'].tolist()

#     def __len__(self):
#         return len(self.unit_sequences)

#     def __getitem__(self, idx):
#         audio, sr = sf.read(self.audio_paths[idx])
#         return {
#             'unit_sequence': torch.LongTensor(self.unit_sequences[idx]),
#             'audio': torch.FloatTensor(audio)
#         }

# def dump_result(output_path, sample_id, pred_wav, sample_rate=16000):
#     sf.write(f"{output_path}/{sample_id}_pred.wav", pred_wav.detach().cpu().numpy(), sample_rate)

# def collate_fn(batch):
#     unit_sequences = [item['unit_sequence'] for item in batch]
#     audio_waveforms = [item['audio'] for item in batch]

#     max_length = max([len(audio) for audio in audio_waveforms])
#     padded_waveforms = [F.pad(audio, (0, max_length - len(audio))) for audio in audio_waveforms]

#     unit_sequences_padded = torch.nn.utils.rnn.pad_sequence(unit_sequences, batch_first=True, padding_value=0)
#     return {
#         'unit_sequences': unit_sequences_padded,
#         'audio_waveforms': torch.stack(padded_waveforms)
#     }

# def train(args):
#     log_dir = args.checkpoint_dir / "logs"
#     log_dir.mkdir(exist_ok=True, parents=True)
#     writer = SummaryWriter(log_dir)

#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     optimizer = optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

#     train_dataset = UnitSequenceDataset(args.train_units_txt, args.train_tsv)
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

#     valid_dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     best_loss = float("inf")
#     global_step = 0

#     for epoch in range(args.epochs):
#         vocoder.train()
#         for i, batch in enumerate(train_dataloader):
#             unit_seq = batch['unit_sequences'].cuda()
#             target_audio = batch['audio_waveforms'].cuda()
#             optimizer.zero_grad()
#             generated_wav = vocoder({"code": unit_seq})["wav"]
#             loss = F.mse_loss(generated_wav, target_audio)
#             loss.backward()
#             optimizer.step()

#             global_step += 1
#             if global_step % args.log_interval == 0:
#                 writer.add_scalar("train/loss", loss.item(), global_step)
#                 logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")

#         scheduler.step()

#         # Validation step
#         vocoder.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for i, batch in enumerate(valid_dataloader):
#                 unit_seq = batch['unit_sequences'].cuda()
#                 target_audio = batch['audio_waveforms'].cuda()
#                 generated_wav = vocoder({"code": unit_seq})["wav"]
#                 loss = F.mse_loss(generated_wav, target_audio)
#                 val_loss += loss.item()
#             val_loss /= len(valid_dataloader)

#         writer.add_scalar("val/loss", val_loss, epoch)
#         logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")

#         if val_loss < best_loss:
#             best_loss = val_loss
#             save_path = args.checkpoint_dir / f"best_checkpoint.pt"
#             torch.save(vocoder.state_dict(), save_path)
#             logger.info(f"New best checkpoint saved at epoch {epoch} with loss {val_loss}")

# def evaluate(args):
#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     results_path = Path(args.results_path)
#     results_path.mkdir(parents=True, exist_ok=True)

#     vocoder.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(dataloader)):
#             unit_seq = batch['unit_sequences'].cuda()
#             pred_wav = vocoder({"code": unit_seq})["wav"]
#             sample_id = Path(batch['audio_waveforms'][0]).stem
#             dump_result(results_path, sample_id, pred_wav)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train-tsv", type=str, default=train_tsv, help="Path to the train TSV file")
#     parser.add_argument("--valid-tsv", type=str, default=valid_tsv, help="Path to the validation TSV file")
#     parser.add_argument("--train-units-txt", type=str, default=train_units_txt, help="Path to the train units text file")
#     parser.add_argument("--valid-units-txt", type=str, default=valid_units_txt, help="Path to the validation units text file")
#     parser.add_argument("--vocoder-checkpoint", type=str, default=vocoder_checkpoint, help="Path to the vocoder checkpoint")
#     parser.add_argument("--vocoder-config", type=str, default=vocoder_config, help="Path to the vocoder config file")
#     parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, help="Directory to save checkpoints")
#     parser.add_argument("--results-path", type=str, default=results_path, help="Directory to save results")
#     parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for fine-tuning")
#     parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=3100, help="Number of epochs for training")
#     parser.add_argument("--log-interval", type=int, default=5, help="Interval for logging training metrics")
#     parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Interval for saving checkpoints")
#     parser.add_argument("--eval", action="store_true", help="Evaluate the model after training")

#     args = parser.parse_args()

#     if not args.eval:
#         train(args)
#     else:
#         evaluate(args)



############################################working###################################
# import argparse
# import json
# import logging
# from pathlib import Path
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
# import soundfile as sf
# import pandas as pd
# from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths to the input files and directories
# train_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_train.tsv'
# valid_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_valid.tsv'
# train_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_train.txt'
# valid_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_valid.txt'
# vocoder_checkpoint = '/home/taruntejaneurips23/s2s_mtp/vocoder/g_00500000'
# vocoder_config = '/home/taruntejaneurips23/s2s_mtp/vocoder/config.json'
# checkpoint_dir = Path('/home/taruntejaneurips23/DEMO/hifi_gan/hifigan/hifigan')
# results_path = Path('/home/taruntejaneurips23/DEMO/evalution_hifigan_audio')

# class UnitSequenceDataset(Dataset):
#     def __init__(self, units_txt, tsv_file):
#         self.unit_sequences = self.load_unit_sequences(units_txt)
#         self.audio_paths = self.load_audio_paths(tsv_file)

#     def load_unit_sequences(self, filepath):
#         with open(filepath, 'r') as f:
#             unit_sequences = [list(map(int, line.strip().split())) for line in f]
#         return unit_sequences

#     def load_audio_paths(self, filepath):
#         df = pd.read_csv(filepath, sep='\t')
#         return df['tgt_audio'].tolist()

#     def __len__(self):
#         return len(self.unit_sequences)

#     def __getitem__(self, idx):
#         audio, sr = sf.read(self.audio_paths[idx])
#         if len(audio.shape) == 2:
#             audio = audio.mean(axis=1)  # Convert to mono by averaging the two channels
#         return {
#             'unit_sequence': torch.LongTensor(self.unit_sequences[idx]),
#             'audio': torch.FloatTensor(audio)
#         }

# def dump_result(output_path, sample_id, pred_wav, sample_rate=16000):
#     sf.write(f"{output_path}/{sample_id}_pred.wav", pred_wav.detach().cpu().numpy(), sample_rate)

# def collate_fn(batch):
#     unit_sequences = [item['unit_sequence'] for item in batch]
#     audio_waveforms = [item['audio'] for item in batch]

#     unit_sequences_padded = torch.nn.utils.rnn.pad_sequence(unit_sequences, batch_first=True, padding_value=0)
#     return {
#         'unit_sequences': unit_sequences_padded,
#         'audio_waveforms': audio_waveforms
#     }

# def train(args):
#     log_dir = args.checkpoint_dir / "logs"
#     log_dir.mkdir(exist_ok=True, parents=True)
#     writer = SummaryWriter(log_dir)

#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     optimizer = optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

#     train_dataset = UnitSequenceDataset(args.train_units_txt, args.train_tsv)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)

#     valid_dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     best_loss = float("inf")
#     global_step = 0

#     for epoch in range(args.epochs):
#         vocoder.train()
#         for i, batch in enumerate(train_dataloader):
#             unit_seq = batch['unit_sequences'].cuda()
#             target_audio = batch['audio_waveforms'][0].cuda()  # Get the first and only audio in the batch

#             optimizer.zero_grad()
#             generated_wav = vocoder({"code": unit_seq})
#             generated_wav = generated_wav.view(-1).requires_grad_(True)

#             print(f"generated_wav: {generated_wav}")
#             print(f"generated_wav shape: {generated_wav.shape}")

#             min_length = min(generated_wav.size(0), target_audio.size(0))
#             loss = F.l1_loss(generated_wav[:min_length], target_audio[:min_length])
#             print(f"loss: {loss}")

#             loss.backward()
#             optimizer.step()

#             global_step += 1
#             if global_step % args.log_interval == 0:
#                 writer.add_scalar("train/loss", loss.item(), global_step)
#                 logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")

#         scheduler.step()

#         # Validation step
#         vocoder.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for i, batch in enumerate(valid_dataloader):
#                 unit_seq = batch['unit_sequences'].cuda()
#                 target_audio = batch['audio_waveforms'][0].cuda()
#                 generated_wav = vocoder({"code": unit_seq}).view(-1)

#                 min_length = min(generated_wav.size(0), target_audio.size(0))
#                 loss = F.l1_loss(generated_wav[:min_length], target_audio[:min_length])
#                 val_loss += loss.item()
#             val_loss /= len(valid_dataloader)

#         writer.add_scalar("val/loss", val_loss, epoch)
#         logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")

#         if val_loss < best_loss:
#             best_loss = val_loss
#             save_path = args.checkpoint_dir / f"best_checkpoint.pt"
#             torch.save(vocoder.state_dict(), save_path)
#             logger.info(f"New best checkpoint saved at epoch {epoch} with loss {val_loss}")

# def evaluate(args):
#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     results_path = Path(args.results_path)
#     results_path.mkdir(parents=True, exist_ok=True)

#     vocoder.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(tqdm(dataloader)):
#             unit_seq = batch['unit_sequences'].cuda()
#             pred_wav = vocoder({"code": unit_seq}).view(-1)
#             sample_id = Path(batch['audio_waveforms'][0]).stem
#             dump_result(results_path, sample_id, pred_wav)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train-tsv", type=str, default=train_tsv, help="Path to the train TSV file")
#     parser.add_argument("--valid-tsv", type=str, default=valid_tsv, help="Path to the validation TSV file")
#     parser.add_argument("--train-units-txt", type=str, default=train_units_txt, help="Path to the train units text file")
#     parser.add_argument("--valid-units-txt", type=str, default=valid_units_txt, help="Path to the validation units text file")
#     parser.add_argument("--vocoder-checkpoint", type=str, default=vocoder_checkpoint, help="Path to the vocoder checkpoint")
#     parser.add_argument("--vocoder-config", type=str, default=vocoder_config, help="Path to the vocoder config file")
#     parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, help="Directory to save checkpoints")
#     parser.add_argument("--results-path", type=str, default=results_path, help="Directory to save results")
#     parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for fine-tuning")
#     parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=3100, help="Number of epochs for training")
#     parser.add_argument("--log-interval", type=int, default=5, help="Interval for logging training metrics")
#     parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Interval for saving checkpoints")
#     parser.add_argument("--eval", action="store_true", help="Evaluate the model after training")

#     args = parser.parse_args()

#     if not args.eval:
#         train(args)
#     else:
#         evaluate(args)


# import argparse
# import json
# import logging
# from pathlib import Path
# import torch
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# from torch.utils.tensorboard import SummaryWriter
# from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
# import soundfile as sf
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Paths to the input files and directories
# train_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_train.tsv'
# valid_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_valid.tsv'
# train_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_train.txt'
# valid_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_valid.txt'
# vocoder_checkpoint = '/home/taruntejaneurips23/s2s_mtp/vocoder/g_00500000'
# vocoder_config = '/home/taruntejaneurips23/s2s_mtp/vocoder/config.json'
# checkpoint_dir = Path('/home/taruntejaneurips23/DEMO/hifi_gan/hifigan/hifigan')
# results_path = Path('/home/taruntejaneurips23/DEMO/evalution_hifigan_audio')

# class UnitSequenceDataset(Dataset):
#     def __init__(self, units_txt, tsv_file):
#         self.unit_sequences = self.load_unit_sequences(units_txt)
#         self.audio_paths = self.load_audio_paths(tsv_file)

#     def load_unit_sequences(self, filepath):
#         with open(filepath, 'r') as f:
#             unit_sequences = [list(map(int, line.strip().split())) for line in f]
#         return unit_sequences

#     def load_audio_paths(self, filepath):
#         df = pd.read_csv(filepath, sep='\t')
#         return df['tgt_audio'].tolist()

#     def __len__(self):
#         return len(self.unit_sequences)

#     def __getitem__(self, idx):
#         try:
#             audio, sr = sf.read(self.audio_paths[idx])
#             if len(audio.shape) == 2:
#                 audio = audio.mean(axis=1)  # Convert to mono by averaging the two channels
#             return {
#                 'unit_sequence': torch.LongTensor(self.unit_sequences[idx]),
#                 'audio': torch.FloatTensor(audio)
#             }
#         except Exception as e:
#             print(f"Skipping audio file {self.audio_paths[idx]} due to error: {e}")
#             return self.__getitem__((idx + 1) % self.__len__())

# def dump_result(output_path, sample_id, pred_wav, sample_rate=16000):
#     sf.write(f"{output_path}/{sample_id}_pred.wav", pred_wav.detach().cpu().numpy(), sample_rate)

# def collate_fn(batch):
#     unit_sequences = [item['unit_sequence'] for item in batch]
#     audio_waveforms = [item['audio'] for item in batch]

#     unit_sequences_padded = torch.nn.utils.rnn.pad_sequence(unit_sequences, batch_first=True, padding_value=0)
#     return {
#         'unit_sequences': unit_sequences_padded,
#         'audio_waveforms': audio_waveforms
#     }

# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     log_dir = args.checkpoint_dir / "logs"
#     log_dir.mkdir(exist_ok=True, parents=True)
#     writer = SummaryWriter(log_dir)

#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     vocoder = vocoder.to(device)

#     optimizer = optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

#     train_dataset = UnitSequenceDataset(args.train_units_txt, args.train_tsv)
#     train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

#     valid_dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     best_loss = float("inf")
#     global_step = 0
#     train_losses = []
#     val_losses = []

#     for epoch in tqdm(range(args.epochs)):
#         vocoder.train()
#         epoch_train_loss = 0
#         for i, batch in enumerate(train_dataloader):
#             unit_seq = batch['unit_sequences'].to(device)
#             target_audio = batch['audio_waveforms'][0].to(device)  # Get the first and only audio in the batch

#             optimizer.zero_grad()
#             generated_wav = vocoder({"code": unit_seq})
#             generated_wav = generated_wav.view(-1).requires_grad_(True)

#             print(f"generated_wav: {generated_wav}")
#             print(f"generated_wav shape: {generated_wav.shape}")

#             min_length = min(generated_wav.size(0), target_audio.size(0))
#             loss = F.l1_loss(generated_wav[:min_length], target_audio[:min_length])
#             print(f"loss: {loss}")

#             loss.backward()
#             optimizer.step()

#             global_step += 1
#             epoch_train_loss += loss.item()
#             if global_step % args.log_interval == 0:
#                 writer.add_scalar("train/loss", loss.item(), global_step)
#                 logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")

#         epoch_train_loss /= len(train_dataloader)
#         train_losses.append(epoch_train_loss)
#         scheduler.step()

#         # Validation step
#         vocoder.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for i, batch in enumerate(valid_dataloader):
#                 unit_seq = batch['unit_sequences'].to(device)
#                 target_audio = batch['audio_waveforms'][0].to(device)
#                 generated_wav = vocoder({"code": unit_seq}).view(-1)

#                 min_length = min(generated_wav.size(0), target_audio.size(0))
#                 loss = F.l1_loss(generated_wav[:min_length], target_audio[:min_length])
#                 val_loss += loss.item()
#             val_loss /= len(valid_dataloader)
#             val_losses.append(val_loss)

#         writer.add_scalar("val/loss", val_loss, epoch)
#         logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")

#         if val_loss < best_loss:
#             best_loss = val_loss
#             save_path = args.checkpoint_dir / f"best_checkpoint.pt"
#             torch.save(vocoder.state_dict(), save_path)
#             logger.info(f"New best checkpoint saved at epoch {epoch} with loss {val_loss}")

#     plt.figure()
#     plt.plot(range(args.epochs), train_losses, label='Train Loss')
#     plt.plot(range(args.epochs), val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(args.checkpoint_dir / 'loss_plot.png')

# def evaluate(args):
#     with open(args.vocoder_config, 'r') as f:
#         vocoder_cfg = json.load(f)

#     vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
#     if torch.cuda.is_available():
#         vocoder = vocoder.cuda()

#     dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     results_path = Path(args.results_path)
#     results_path.mkdir(parents=True, exist_ok=True)

#     vocoder.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(dataloader):
#             unit_seq = batch['unit_sequences'].cuda()
#             pred_wav = vocoder({"code": unit_seq}).view(-1)
#             sample_id = Path(batch['audio_waveforms'][0]).stem
#             dump_result(results_path, sample_id, pred_wav)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train-tsv", type=str, default=train_tsv, help="Path to the train TSV file")
#     parser.add_argument("--valid-tsv", type=str, default=valid_tsv, help="Path to the validation TSV file")
#     parser.add_argument("--train-units-txt", type=str, default=train_units_txt, help="Path to the train units text file")
#     parser.add_argument("--valid-units-txt", type=str, default=valid_units_txt, help="Path to the validation units text file")
#     parser.add_argument("--vocoder-checkpoint", type=str, default=vocoder_checkpoint, help="Path to the vocoder checkpoint")
#     parser.add_argument("--vocoder-config", type=str, default=vocoder_config, help="Path to the vocoder config file")
#     parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, help="Directory to save checkpoints")
#     parser.add_argument("--results-path", type=str, default=results_path, help="Directory to save results")
#     parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for fine-tuning")
#     parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
#     parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
#     parser.add_argument("--log-interval", type=int, default=5, help="Interval for logging training metrics")
#     parser.add_argument("--checkpoint-interval", type=int, default=5, help="Interval for saving checkpoints")
#     parser.add_argument("--eval", action="store_true", help="Evaluate the model after training")

#     args = parser.parse_args()

#     if not args.eval:
#         train(args)
#     else:
#         evaluate(args)




import argparse
import json
import logging
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchaudio.transforms import MelSpectrogram
import numpy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to the input files and directories
train_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_train.tsv'
valid_tsv = '/home/taruntejaneurips23/s2s_mtp/Data_root/tgt_unit_valid.tsv'
train_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_train.txt'
valid_units_txt = '/home/taruntejaneurips23/s2s_mtp/Data_root/unit_valid.txt'
vocoder_checkpoint = '/home/taruntejaneurips23/s2s_mtp/vocoder/g_00500000'
vocoder_config = '/home/taruntejaneurips23/s2s_mtp/vocoder/config.json'
checkpoint_dir = Path('/home/taruntejaneurips23/DEMO/hifi_gan/hifigan/hifigan')
results_path = Path('/home/taruntejaneurips23/DEMO/evalution_hifigan_audio')

class UnitSequenceDataset(Dataset):
    def __init__(self, units_txt, tsv_file):
        self.unit_sequences = self.load_unit_sequences(units_txt)
        self.audio_paths = self.load_audio_paths(tsv_file)

    def load_unit_sequences(self, filepath):
        with open(filepath, 'r') as f:
            unit_sequences = [list(map(int, line.strip().split())) for line in f]
        return unit_sequences

    def load_audio_paths(self, filepath):
        df = pd.read_csv(filepath, sep='\t')
        return df['tgt_audio'].tolist()

    def __len__(self):
        return len(self.unit_sequences)

    def __getitem__(self, idx):
        try:
            audio, sr = sf.read(self.audio_paths[idx])
            if len(audio.shape) == 2:
                audio = audio.mean(axis=1)  # Convert to mono by averaging the two channels
            return {
                'unit_sequence': torch.LongTensor(self.unit_sequences[idx]),
                'audio': torch.FloatTensor(audio)
            }
        except Exception as e:
            print(f"Skipping audio file {self.audio_paths[idx]} due to error: {e}")
            return self.__getitem__((idx + 1) % self.__len__())

def dump_result(output_path, sample_id, pred_wav, sample_rate=16000):
    sf.write(f"{output_path}/{sample_id}_pred.wav", pred_wav.detach().cpu().numpy(), sample_rate)

def collate_fn(batch):
    unit_sequences = [item['unit_sequence'] for item in batch]
    audio_waveforms = [item['audio'] for item in batch]

    unit_sequences_padded = torch.nn.utils.rnn.pad_sequence(unit_sequences, batch_first=True, padding_value=0)
    return {
        'unit_sequences': unit_sequences_padded,
        'audio_waveforms': audio_waveforms
    }

def compute_mel_spectrogram(folder_path):
    mel_spectrogram = np.load(folder_path)
    return mel_spectrogram

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = args.checkpoint_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir)

    with open(args.vocoder_config, 'r') as f:
        vocoder_cfg = json.load(f)

    vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
    vocoder = vocoder.to(device)

    optimizer = optim.AdamW(vocoder.parameters(), lr=args.learning_rate, betas=(0.8, 0.99), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    train_dataset = UnitSequenceDataset(args.train_units_txt, args.train_tsv)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    valid_dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    best_loss = float("inf")
    global_step = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(args.epochs)):
        vocoder.train()
        epoch_train_loss = 0
        for i, batch in enumerate(train_dataloader):
            unit_seq = batch['unit_sequences'].to(device)
            target_audio = batch['audio_waveforms'][0].to(device)  # Get the first and only audio in the batch

            optimizer.zero_grad()
            generated_wav = vocoder({"code": unit_seq})
            generated_wav = generated_wav.view(-1).requires_grad_(True)

            print(f"generated_wav: {generated_wav}")
            print(f"generated_wav shape: {generated_wav.shape}")

            min_length = min(generated_wav.size(0), target_audio.size(0))
            generated_wav = generated_wav[:min_length]
            target_audio = target_audio[:min_length]

            generated_mel = compute_mel_spectrogram(generated_wav, device)
            target_mel = compute_mel_spectrogram(target_audio, device)

            loss = F.l1_loss(generated_mel, target_mel)
            print(f"loss: {loss}")

            loss.backward()
            optimizer.step()
            
            global_step += 1
            epoch_train_loss += loss.item()
            if global_step % args.log_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                logger.info(f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item()}")

        epoch_train_loss /= len(train_dataloader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        # Validation step
        vocoder.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(valid_dataloader):
                unit_seq = batch['unit_sequences'].to(device)
                target_audio = batch['audio_waveforms'][0].to(device)
                generated_wav = vocoder({"code": unit_seq}).view(-1)

                min_length = min(generated_wav.size(0), target_audio.size(0))
                generated_wav = generated_wav[:min_length]
                target_audio = target_audio[:min_length]
          
                generated_mel = compute_mel_spectrogram(generated_wav, device)
                target_mel = compute_mel_spectrogram(target_audio, device)

                loss = F.l1_loss(generated_mel, target_mel)
                val_loss += loss.item()
            val_loss /= len(valid_dataloader)
            val_losses.append(val_loss)

        writer.add_scalar("val/loss", val_loss, epoch)
        logger.info(f"Epoch: {epoch}, Validation Loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_path = args.checkpoint_dir / f"best_checkpoint.pt"
            torch.save(vocoder.state_dict(), save_path)
            logger.info(f"New best checkpoint saved at epoch {epoch} with loss {val_loss}")

    plt.figure()
    plt.plot(range(args.epochs), train_losses, label='Train Loss')
    plt.plot(range(args.epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.checkpoint_dir / 'loss_plot.png')

def evaluate(args):
    with open(args.vocoder_config, 'r') as f:
        vocoder_cfg = json.load(f)

    vocoder = CodeHiFiGANVocoder(args.vocoder_checkpoint, vocoder_cfg)
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()

    dataset = UnitSequenceDataset(args.valid_units_txt, args.valid_tsv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    vocoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            unit_seq = batch['unit_sequences'].cuda()
            pred_wav = vocoder({"code": unit_seq}).view(-1)
            sample_id = Path(batch['audio_waveforms'][0]).stem
            dump_result(results_path, sample_id, pred_wav)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tsv", type=str, default=train_tsv, help="Path to the train TSV file")
    parser.add_argument("--valid-tsv", type=str, default=valid_tsv, help="Path to the validation TSV file")
    parser.add_argument("--train-units-txt", type=str, default=train_units_txt, help="Path to the train units text file")
    parser.add_argument("--valid-units-txt", type=str, default=valid_units_txt, help="Path to the validation units text file")
    parser.add_argument("--vocoder-checkpoint", type=str, default=vocoder_checkpoint, help="Path to the vocoder checkpoint")
    parser.add_argument("--vocoder-config", type=str, default=vocoder_config, help="Path to the vocoder config file")
    parser.add_argument("--checkpoint-dir", type=Path, default=checkpoint_dir, help="Directory to save checkpoints")
    parser.add_argument("--results-path", type=str, default=results_path, help="Directory to save results")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--log-interval", type=int, default=5, help="Interval for logging training metrics")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Interval for saving checkpoints")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model after training")

    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        evaluate(args)
