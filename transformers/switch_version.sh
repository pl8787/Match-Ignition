
if [ $1 ]; then
    pip uninstall -y transformers
    python setup.py install
else
    pip uninstall -y transformers
    pip install transformers==3.4.0
fi
