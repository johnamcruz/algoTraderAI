#python3 algoTrader.py --backtest --backtest_data data/ES_continuous_5min.csv --contract CON.F.US.EP.Z25 --risk_amount 300 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --simulation-days 10

#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.ENQ.Z25 --risk_amount 300 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --simulation-days 10 --high_conf_multiplier 2.0 --max_loss 800 --profit_target 6000

#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --simulation-days 30 --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000

# Bear Market (2022 NQ Crash)
#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000 --start-date 2022-01-01 --end-date 2022-10-15

# Recovery / Rebound (2023)
python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000 --start-date 2023-01-01 --end-date 2023-12-31

# Aug 2024 Selloff
#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000 --start-date 2024-07-15 --end-date 2024-09-15

# High-Vol Chop (2023 Banking Crisis)
#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000 --start-date 2023-03-01 --end-date 2023-05-31

# Out-of-Sample Control (2021)
#python3 algoTrader.py --backtest --backtest_data data/NQ_continuous_5min.csv --contract CON.F.US.MNQ.M26 --risk_amount 200 --max_contracts 5 --entry_conf 0.70 --model models/cisd_ote_hybrid_v5_1.onnx --strategy "cisd-ote" --high_conf_multiplier 2.0 --max_loss 400 --profit_target 6000 --start-date 2021-01-01 --end-date 2021-12-31