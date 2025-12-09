# Deployment Guide

## GitHub Setup

1. **Initialize Git Repository**
```bash
git init
git add .
git commit -m "Initial commit - Phishing URL Detection App"
```

2. **Create GitHub Repository**
- Go to GitHub.com
- Click "New Repository"
- Name it "phishing-url-detection" 
- Don't initialize with README (we already have one)
- Click "Create Repository"

3. **Push to GitHub**
```bash
git remote add origin https://github.com/YOUR_USERNAME/phishing-url-detection.git
git branch -M main
git push -u origin main
```

## Render Deployment

1. **Connect GitHub to Render**
- Go to [render.com](https://render.com)
- Sign up/Login with GitHub
- Click "New +" → "Web Service"

2. **Configure Web Service**
- **Repository**: Select your phishing-url-detection repo
- **Branch**: main
- **Root Directory**: leave blank
- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt && python train_model.py`
- **Start Command**: `gunicorn app:app`

3. **Environment Variables** (Optional)
- You can add any environment variables if needed
- For this app, default settings work fine

4. **Deploy**
- Click "Create Web Service"
- Wait for deployment (usually 2-5 minutes)
- Your app will be available at: `https://your-app-name.onrender.com`

## Important Notes

- **First deployment** takes longer due to model training
- **Free tier** limitations: app may sleep after 15 minutes of inactivity
- **Model file** is automatically created during deployment
- **Dataset** (phishing.csv) must be included in the repository

## Troubleshooting

### Common Issues:

1. **Build fails**: Check requirements.txt for compatible versions
2. **Model not loading**: Ensure train_model.py runs successfully  
3. **App crashes**: Check logs in Render dashboard
4. **Slow response**: Normal for free tier after sleep

### Local Testing:
```bash
# Test locally before deploying
python train_model.py  # Train model
python app.py          # Run app
# Visit http://localhost:5000
```

## File Structure for Deployment:
```
phishing-url-detection/
├── app.py              # Main Flask app
├── feature.py          # Feature extraction
├── train_model.py      # Model training
├── requirements.txt    # Dependencies
├── Procfile           # Gunicorn config
├── runtime.txt        # Python version
├── build.sh           # Build script
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
├── phishing.csv       # Training data
├── static/            # CSS files
│   └── styles.css
├── templates/         # HTML templates
│   └── index.html
└── pickle/            # Model directory
    └── model.pkl      # Trained model (auto-generated)
```

## Post-Deployment:

1. Test the deployed app
2. Share the URL with others  
3. Monitor performance in Render dashboard
4. Update repository for any changes (auto-deploys)