#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.8 --stop_pts 8.0 --target_pts 16.0 --model models/model_trendpullback_3min_v2.onnx --scaler models/scalar_trendpullback_3min_v2.pkl --strategy trend_pullback2 --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_ES1_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.9 --stop_pts 8.0 --target_pts 16.0 --model models/model_pivot_action_3min_v3_final.onnx --scaler models/scalar_pivot_action_3min_v3_final.pkl --strategy "pivot_action" --simulation-days 10

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_ES1_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.9 --stop_pts 8.0 --target_pts 12.0 --model models/model_supertrend_pullback_v3.10.onnx --scaler models/scaler_supertrend_pullback_v3.10.pkl --strategy "supertrend" --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.9 --stop_pts 8.0 --target_pts 16.0 --model models/model_supertrend_pullback_v3.10.onnx --scaler models/scaler_supertrend_pullback_v3.10.pkl --strategy "supertrend" --simulation-days 30

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.85 --stop_pts 8.0 --target_pts 16.0 --model models/ema_pullback_v1.onnx --scaler models/ema_pullback_v1_scaler.pkl --strategy "ema_pullback" --simulation-days 30

python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.8 --stop_pts 8.0 --target_pts 16.0 --model models/supertrend_pullback_v4.8.onnx --scaler models/supertrend_pullback_v4.8_scaler.pkl --strategy "supertrend2" --simulation-days 30
