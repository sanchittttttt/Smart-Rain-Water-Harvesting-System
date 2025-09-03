# 💧 RainFlow - Smart Rainwater Harvesting Management System

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-orange)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🌟 Overview

RainFlow is an AI-powered rainwater harvesting management system that combines real-time NASA weather data, LSTM forecasting, genetic algorithm optimization, and tank simulation to provide intelligent water management solutions.

## ✨ Features

- **🌤️ Real-time NASA POWER API Integration** - Live rainfall data from NASA
- **🔮 LSTM Neural Network Forecasting** - 7-day rainfall predictions
- **🧬 Genetic Algorithm Optimization** - Smart water usage planning
- **💧 Tank Simulation** - Realistic water level modeling
- **📊 Interactive Dashboard** - Beautiful Streamlit interface
- **📍 Location-based Analysis** - Customizable coordinates
- **📈 Data Visualization** - Plotly charts and graphs

## 🚀 Live Demo

**Coming Soon!** - Deploying to Streamlit Cloud

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/rainflow.git
cd rainflow

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run dashboard.py
```

## 📦 Dependencies

- **Streamlit** - Web dashboard framework
- **TensorFlow/Keras** - LSTM neural network
- **Pandas & NumPy** - Data manipulation
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning utilities
- **Requests** - API calls to NASA POWER

## 🎯 Usage

1. **Set Location** - Enter your latitude and longitude
2. **Configure System** - Set tank capacity and usage parameters
3. **Generate Forecast** - Get 7-day rainfall predictions
4. **Optimize Usage** - AI-powered water consumption planning
5. **Simulate Tank** - Visualize water levels over time

## 🔧 Configuration

### Environment Variables
```bash
# NASA POWER API (no key required - public API)
# All other settings configurable via dashboard
```

### Model Parameters
- **LSTM Sequence Length**: 30 days
- **Forecast Horizon**: 7 days
- **Genetic Algorithm Generations**: 100
- **Population Size**: 50

## 📁 Project Structure

```
rainflow/
├── dashboard.py              # Main Streamlit dashboard
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python dependencies
├── README.md                # This file
├── src/                     # Source code
│   ├── fetch_live_data_nasa.py    # NASA API integration
│   ├── predict_live.py            # LSTM forecasting
│   ├── ga_optimization.py         # Genetic algorithm
│   └── tank_simulation.py         # Tank simulation
├── models/                  # Trained LSTM models
├── data/                    # Historical data
└── outputs/                 # Generated plots and results
```

## 🌐 API Integration

### NASA POWER API
- **Endpoint**: https://power.larc.nasa.gov/api/temporal/daily/
- **Data**: Daily rainfall, temperature, humidity
- **Coverage**: Global, 1981-present
- **Update**: Daily
- **Authentication**: None required (public API)

## 🤖 AI Models

### LSTM Neural Network
- **Architecture**: Long Short-Term Memory
- **Input**: 30 days of rainfall data
- **Output**: 7-day forecast
- **Training**: Historical NASA data
- **Accuracy**: Optimized for rainfall patterns

### Genetic Algorithm
- **Objective**: Minimize water shortage
- **Constraints**: Tank capacity, daily usage
- **Population**: 50 individuals
- **Generations**: 100 iterations
- **Selection**: Tournament selection

## 📊 Performance Metrics

- **Forecast Accuracy**: RMSE-based evaluation
- **Optimization Score**: Water shortage minimization
- **Simulation Accuracy**: Real-world tank behavior modeling

## 🔮 Future Enhancements

- [ ] Multi-location support
- [ ] Advanced weather models
- [ ] Mobile app version
- [ ] IoT sensor integration
- [ ] Machine learning model retraining
- [ ] Export functionality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA POWER Project** - Weather data API
- **Streamlit Team** - Web framework
- **TensorFlow Community** - Machine learning framework
- **Open Source Contributors** - Various libraries and tools


---
**Made with ❤️ for sustainable water management**
