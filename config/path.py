import os  

try:
    IN_COLAB = "google.colab" in str(get_ipython())
except NameError:
    IN_COLAB = False
    
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = "/content/drive/MyDrive/clip_features"
else:
    BASE_DIR = os.path.join(os.getcwd(), "clip_features")

os.makedirs(BASE_DIR, exist_ok=True)