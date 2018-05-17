# Transfer-Adder
Transferable Adder from previous digits adder
## Introduction
Simple experiment is carried out to test the performance of transfer learning.

Two versions of adder have be programmed: **Original** ver.& **Transfer** ver.

Weights of original ver. will be initialed randomly. 

On the other hand, weights of transfer ver. will be initialed by previous digits of adder.

If you want to train a transfer 4-digits adder, original 3-digits adder should be trained before.

Remark: Training will be stopped when validation accuracy meets `0.9`.
## Usage

`python3 Ori_Addition.py <digits>` 

`python3 Transfer_Addition.py <digits>` 

digits(int): Number of digits adder you perfer to train

`python3 ori.py`

`ori.py` script will train 3-7 digits original adder automatically.

`python3 transfer.py`

`transfer.py` script will train 4-7 digits transfered adder automatically.

