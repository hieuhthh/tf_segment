import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

url = "https://drive.google.com/file/d/1z48bsJftdp4akAlWOziqt6032huYYN9k/view?usp=sharing"
output =  f"{des}/data.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)