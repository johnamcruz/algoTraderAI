#python3 algoTrader.py --backtest --backtest_data data/ES_continuous_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.75 --stop_pts 8.0 --target_pts 32.0 --model models/supertrend_pullback_v4.9.onnx --scaler models/supertrend_pullback_v4.9_scaler.pkl --strategy "supertrend3" --simulation-days 10

#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.75 --stop_pts 40.0 --target_pts 120.0  --model models/supertrend_pullback_v4.9.onnx --scaler models/supertrend_pullback_v4.9_scaler.pkl --strategy "supertrend3" --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/ES_continuous_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.4 --stop_pts 8.0 --target_pts 32.0 --model models/multihorizon_v3_4.onnx --scaler models/multihorizon_v3_4_scaler.pkl --strategy "multihorizon" --simulation-days 30

#python3 algoTrader.py --backtest --backtest_data data/ES_continuous_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.65 --stop_pts 8.0 --target_pts 32.0 --model models/multi_tf_transformer_v6_3_6.onnx --scaler models/multi_tf_transformer_v6_3_6_scaler.pkl --strategy "mtf_directional" --simulation-days 3

#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.4 --stop_pts 12.0 --target_pts 48.0 --model models/multihorizon_v3_6.onnx --scaler models/multihorizon_v3_6_scaler.pkl --strategy "multihorizon" --simulation-days 10

#python3 algoTrader.py --backtest --backtest_data data/ES_continuous_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.65 --stop_pts 8.0 --target_pts 32.0 --model models/ema_transformer_v1_2_fixed.onnx --scaler models/ema_transformer_v1_1_scaler.pkl --strategy "ema" --simulation-days 2

python3 algoTrader.py --backtest --backtest_data data/ES_continuous_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.65 --stop_pts 8.0 --target_pts 32.0 --model models/flag_pennant_v2_4.onnx --scaler models/flag_pennant_v2_4_scaler.pkl --strategy "flag" --simulation-days 30