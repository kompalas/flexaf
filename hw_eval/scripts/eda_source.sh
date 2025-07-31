export SNPSLMD_QUEUE=true

# LICENSE_SERVER="150.140.141.106"
# export SNPSLMD_LICENSE_FILE="27020@$LICENSE_SERVER"
# export LM_LICENSE_FILE="$LM_LICENSE_FILE:$SNPSLMD_LICENSE_FILE"


# LICENSE_SERVER="flex.ceid.upatras.gr"
LICENSE_SERVER="flexclass.ceid.upatras.gr"

export SNPSLMD_LICENSE_FILE="27002@$LICENSE_SERVER"
export LM_LICENSE_FILE="$LM_LICENSE_FILE:27000$SLICENSE_SERVER"

# export SNPS_SYN="/usr/local/eda/synopsys/dc/syn/T-2022.03-SP5"
export SNPS_SYN="/usr/local/eda/synopsys/dc/syn/S-2021.06-SP2"
export SYNOPSYS_SYN_ROOT="${SNPS_SYN}"
export PATH="${PATH}:${SNPS_SYN}/bin"


export SNPS_PT="/usr/local/eda/synopsys/pt/prime/T-2022.03-SP5"
export PATH="${PATH}:${SNPS_PT}/bin"


export SNPS_LC="/usr/local/eda/synopsys/lc/lc/T-2022.03-SP5"
export PATH="${PATH}:${SNPS_LC}/bin"
export SYNOPSYS_LC_ROOT="${SNPS_LC}/bin"


export VCS_HOME="/usr/local/eda/synopsys/vcs/vcs/T-2022.06-SP1-1"
export VCS_ARCH_OVERRIDE="linux"
export PATH="${PATH}:${VCS_HOME}/bin"

export VERDI_HOME="/usr/local/eda/synopsys/verdi/verdi/T-2022.06-SP1-1"
export PATH="${PATH}:${VERDI_HOME}/bin"
