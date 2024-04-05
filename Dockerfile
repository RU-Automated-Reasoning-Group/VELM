FROM --platform=linux/x86_64 ubuntu:22.04

COPY VELM /

CMD apt install apt install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc g++ libopenmpi-dev

