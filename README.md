# dqn-rust

todo: ADD REQUIREMENTS
reimplementation of deep q network

```bash
sudo apt install unrar -y

pip install gym
pip install gym[atari]

wget http://www.atarimania.com/roms/Roms.rar
tar -xvf Roms.rar
mkdir roms
unrar e Roms.rar roms
cd roms
unzip ROMS.zip
ale-import-roms ROMS
rm Roms.rar

cd ../..
pip install -r requirements.txt
python3 src/main.py
```
