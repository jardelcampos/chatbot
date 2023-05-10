#activate_this = '/home/ubuntu/odb/bin/activate_this.py'
activate_this = '/home/jardelsewo.seed/Documentos/venv/bin/activate_this.py'

with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

import sys
sys.path.insert(0, '/var/www/flaskapp')
from flaskapp import app as application
