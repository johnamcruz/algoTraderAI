# 🧠 LSTM Futures Trader (ProjectX)

A lightweight console application that automatically trades futures contracts on the **ProjectX** platform.  
It uses an **LSTM (Long Short-Term Memory)** model to detect market trends and avoid chop, executing trades based on the model’s signals with built-in risk management.

---

## ⚙️ Overview

**LSTM Futures Trader** is designed to:
- Detect trend direction using a trained LSTM model  
- Automatically enter and exit futures positions  
- Apply default risk management (stop-loss, take-profit, position sizing)  
- Run continuously from the console with minimal setup  

The goal is to **stay in trending markets** and **avoid sideways noise** — maximizing trend capture efficiency.

---

## 🏗️ How It Works

1. The app streams live futures data from **ProjectX**.  
2. Data is processed and fed into the LSTM model.  
3. The model classifies the current market state:
   - `UPTREND`
   - `DOWNTREND`
   - `CHOP` (no trade zone)
4. The trade engine executes or exits positions accordingly using default risk parameters.


