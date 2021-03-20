# Health  Monitoring System for Autonomous Vehicles using Dynamic Bayesian Networks for Diagnosis and Prognosis

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science

## Introduction

HMS_Autonomous_Vehicle is an implementation of a Fault Detection and Diagnosis (FDD) and Prognosis System (PS) for Autonomous Vehicles, which monitor the GPS sensor and Lateral and Longitudinal Controllers, using Dynamic Bayesian Network (DBN).  It also provides the datasets used for learning the DBN parameters, which were collected using the vehicle CaRINA 2 (Intelligent Robotic Car for Autonomous Navigation) [3].


## License

Apache License 2.0

## Citation
  
BibTeX:

    @article{gomes2021health,
      title={Health Monitoring System for Autonomous Vehicles using Dynamic Bayesian Networks for Diagnosis and Prognosis},
      author={Gomes, Iago Pach{\^e}co and Wolf, Denis Fernando},
      journal={Journal of Intelligent \& Robotic Systems},
      volume={101},
      number={1},
      pages={1--21},
      year={2021},
      publisher={Springer}
    }
    
    
## Usage

1) Download the datasets using the script <i>download.sh</i> inside the folder <i>dataset</i>

2) (optional) if you want to use the data with Data Imputation run the code <i>data_imputation.py</i> inside the folder <i>dataset</i> 

3) The codes of each DBN model are inside the folder <i>models</id>

4) To run the model you need to download the SMILE library from the link: https://www.bayesfusion.com/downloads/

5) You can add the library (SMILE) folder in the PYTHONPATH enviroment variable so that python code finds it. 

## References

[1] GOMES, Iago Pachêco; WOLF, Denis Fernando. Health  Monitoring System for Autonomous Vehicles using Dynamic Bayesian Networks for Diagnosis and Prognosis. In: ICAR 2019 Special Issue - Journal of Intelligent & Robotic Systems. 2020.  

[2] GOMES, Iago Pachêco; WOLF, Denis Fernando. A Health Monitoring System with Hybrid Bayesian Network for Autonomous Vehicle. In: 2019 19th International Conference on Advanced Robotics (ICAR). IEEE, 2019. p. 260-265.

[3] FERNANDES, Leandro C. et al. Intelligent robotic car for autonomous navigation: Platform and system architecture. In: 2012 Second Brazilian Conference on Critical Embedded Systems. IEEE, 2012. p. 12-17.

## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'
