# dqn-rust

todo: ADD REQUIREMENTS
reimplementation of deep q network

```bash
sudo apt install unrar swig -y

pip install -r requirements.txt

wget http://www.atarimania.com/roms/Roms.rar
tar -xvf Roms.rar
mkdir roms
unrar e Roms.rar roms
cd roms
unzip ROMS.zip
ale-import-roms ROMS

cd ..
rm Roms.rar
python3 -m src.main
```
