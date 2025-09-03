# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - Your code must be on GitHub
2. **Streamlit Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Working Project** - Your RainFlow dashboard should run locally

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Verify your repository structure:**
   ```
   rainflow/
   ├── dashboard.py          # Main dashboard file
   ├── requirements.txt      # Dependencies
   ├── README.md            # Project documentation
   ├── .gitignore           # Git ignore file
   ├── .streamlit/          # Streamlit config
   │   └── config.toml
   ├── src/                 # Source code
   ├── models/              # AI models
   └── data/                # Data files
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app:**

   **Repository:** `yourusername/rainflow`
   
   **Branch:** `main`
   
   **Main file path:** `dashboard.py`
   
   **App URL:** `rainflow` (or your preferred name)

5. **Click "Deploy!"**

### Step 3: Monitor Deployment

1. **Watch the build logs** for any errors
2. **Common issues and solutions:**

   **❌ Import errors:**
   - Check `requirements.txt` has all dependencies
   - Ensure file paths are correct

   **❌ Model loading errors:**
   - Verify model files are in the repository
   - Check file permissions

   **❌ API errors:**
   - NASA POWER API is public (no key needed)
   - Check internet connectivity

## 🎯 Post-Deployment

### 1. **Test Your App**
- Navigate through all features
- Test location input
- Generate forecasts
- Run optimizations
- Check tank simulation

### 2. **Custom Domain (Optional)**
- Go to app settings
- Add custom domain
- Update DNS records

### 3. **Monitor Performance**
- Check Streamlit Cloud analytics
- Monitor API usage
- Track user engagement

## 🔍 Troubleshooting

### **Build Fails**
```bash
# Check requirements.txt
pip install -r requirements.txt

# Test locally first
streamlit run dashboard.py
```

### **App Crashes**
- Check Streamlit Cloud logs
- Verify all imports work
- Test with minimal code first

### **Slow Performance**
- Optimize model loading
- Reduce data processing
- Use caching where possible

## 📱 Sharing Your App

### **Public URL**
Your app will be available at:
```
https://your-app-name-yourusername.streamlit.app
```

### **Embed in Websites**
```html
<iframe 
  src="https://your-app-name-yourusername.streamlit.app" 
  width="100%" 
  height="800px">
</iframe>
```

### **Social Media**
- Share the direct link
- Create demo videos
- Post screenshots

## 🚀 Advanced Configuration

### **Environment Variables**
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
```

### **Custom Themes**
```toml
[theme]
primaryColor = "#0066cc"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### **Performance Optimization**
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
```

## 📊 Analytics & Monitoring

### **Streamlit Cloud Dashboard**
- View app statistics
- Monitor usage patterns
- Track performance metrics

### **Custom Analytics**
- Google Analytics integration
- User behavior tracking
- Performance monitoring

## 🔄 Updates & Maintenance

### **Automatic Updates**
- Push to GitHub main branch
- Streamlit Cloud auto-deploys
- No manual intervention needed

### **Version Control**
- Use semantic versioning
- Tag releases
- Maintain changelog

## 🎉 Success!

Once deployed, your RainFlow app will be:
- ✅ **Publicly accessible** worldwide
- ✅ **Automatically updated** on code changes
- ✅ **Scalable** for multiple users
- ✅ **Professional** appearance
- ✅ **Mobile responsive**

---

**Need help?** Check the [Streamlit documentation](https://docs.streamlit.io) or create an issue in your repository. 