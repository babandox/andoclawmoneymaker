"""Full training pipeline: Phases 1-3 in sequence with exit criteria checks.

Usage:
    python -m radiant_seer.training.run_pipeline
    python -m radiant_seer.training.run_pipeline --epochs-encoder 50 --epochs-predictor 40
"""

from __future__ import annotations

import argparse

import torch
from loguru import logger

from radiant_seer.configuration.settings import SeerConfig
from radiant_seer.training.evaluate import run_full_evaluation
from radiant_seer.training.train_encoder import train_encoder
from radiant_seer.training.train_predictor import train_predictor
from radiant_seer.training.train_reward import train_reward_decoder


def main(args: argparse.Namespace | None = None) -> None:
    parser = argparse.ArgumentParser(description="Radiant Seer training pipeline")
    parser.add_argument("--epochs-encoder", type=int, default=100)
    parser.add_argument("--epochs-predictor", type=int, default=80)
    parser.add_argument("--epochs-decoder", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    args = args or parser.parse_args()

    config = SeerConfig()
    save_dir = config.project_root / "data" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = save_dir / "encoder_vicreg.pt"
    predictor_path = save_dir / "predictor.pt"
    decoder_path = save_dir / "outcome_decoder.pt"

    if args.eval_only:
        logger.info("Loading saved models for evaluation...")
        from radiant_seer.intelligence.causal_predictor import CausalPredictor
        from radiant_seer.intelligence.multimodal_encoder import MultimodalEncoder
        from radiant_seer.planning.reward_module import OutcomeDecoder

        encoder = MultimodalEncoder(
            news_dim=config.news_embedding_dim,
            macro_dim=config.macro_feature_count,
            latent_dim=config.latent_dim,
        )
        predictor = CausalPredictor(latent_dim=config.latent_dim)
        decoder = OutcomeDecoder(latent_dim=config.latent_dim)

        encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
        predictor.load_state_dict(torch.load(predictor_path, weights_only=True))
        decoder.load_state_dict(torch.load(decoder_path, weights_only=True))
    else:
        # Phase 1: Train encoder
        logger.info("=" * 60)
        logger.info("PHASE 1: Training Multimodal Encoder with VICReg")
        logger.info("=" * 60)
        encoder, enc_history = train_encoder(
            config=config,
            n_episodes=args.episodes,
            epochs=args.epochs_encoder,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=encoder_path,
        )

        # Phase 2: Train predictor
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Training Causal Predictor")
        logger.info("=" * 60)
        predictor, pred_history = train_predictor(
            encoder=encoder,
            config=config,
            n_episodes=args.episodes,
            epochs=args.epochs_predictor,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=predictor_path,
        )

        # Phase 3: Train outcome decoder
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Training Outcome Decoder")
        logger.info("=" * 60)
        decoder, dec_history = train_reward_decoder(
            encoder=encoder,
            config=config,
            n_episodes=1000,
            epochs=args.epochs_decoder,
            save_path=decoder_path,
        )

    # Evaluate all phases
    logger.info("\n")
    run_full_evaluation(encoder, predictor, decoder, config)

    # Save OOD detector baseline
    logger.info("\nFitting OOD detector on training distribution...")
    from radiant_seer.data_swarm.synthetic import SyntheticCivStateGenerator
    from radiant_seer.main_seer import OODDetector

    device = config.device
    encoder = encoder.to(device).eval()
    gen = SyntheticCivStateGenerator(seed=42)
    dataset = gen.generate_dataset(n_episodes=200, episode_length=30)

    with torch.no_grad():
        z_train = encoder(
            dataset.news[:, -1].to(device),
            dataset.macro[:, -1].to(device),
            dataset.sentiment[:, -1].to(device),
        )

    ood = OODDetector(threshold=config.mahalanobis_threshold)
    ood.fit(z_train)
    torch.save(
        {"mean": ood._mean, "inv_cov": ood._inv_cov, "n_samples": len(z_train)},
        save_dir / "ood_baseline.pt",
    )
    logger.info("OOD baseline saved.")

    logger.info("\nPipeline complete. Models saved to: " + str(save_dir))


if __name__ == "__main__":
    main()
