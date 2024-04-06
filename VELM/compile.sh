mkdir -p results/programs

cd POLAR_Tool/examples

# cd acc_spice && make clean && make && cd ..
cd pendulum && make clean && make && cd ..
cd obstacle_mid && make clean && make && cd ..
cd cartpole && make clean && make && cd ..

cd obstacle && make clean && make && cd ..
cd road_2d && make clean && make && cd ..
cd car_racing && make clean && make && cd ..
cd cartpole_move && make clean && make && cd ..

cd cartpole_swing && make clean && make && cd ..
cd lalo && make clean && make && cd ..

