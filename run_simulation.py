#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.8 --stop_atr 1.0 --target_atr 2.0 --model models/model_trendpullback_3min_v2.onnx --scaler models/scalar_trendpullback_3min_v2.pkl --strategy trend_pullback2 --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_ES1_3min.csv --contract CON.F.US.EP.Z25 --size 1 --entry_conf 0.25 --stop_atr 1.0 --target_atr 2.0 --model models/model_pivot_action_3min_v3_final.onnx --scaler models/scalar_pivot_action_3min_v3_final.pkl --strategy "pivot_action" --simulation-days 1

python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.85 --stop_atr 1.0 --target_atr 2.0 --model models/model_supertrend_3min_v38.onnx --scaler models/scalar_supertrend_3min_v38.pkl --strategy "supertrend" --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.0 --stop_atr 1.0 --target_atr 2.0 --model models/model_trendpullback_3min_v2.onnx --scaler models/scalar_trendpullback_3min_v2.pkl --strategy "trend_pullback2" --simulation-days 5

#python3 algoTrader.py --backtest --backtest_data data/CME_MINI_NQ1_3min.csv --contract CON.F.US.ENQ.Z25 --size 1 --entry_conf 0.0 --stop_atr 1.0 --target_atr 2.0 --model models/model_vwap_meanreversion_3min_v3.onnx --scaler models/scalar_vwap_meanreversion_3min_v3.pkl --strategy "vwap" --simulation-days 5