#set -x

function UpdateCopyrights () {
directory=$1
spaces="                                                    "
cd ${directory}
git ls-tree -r --name-status HEAD | while read file_name; do
  last_update_year=$(git log -1 --date=short --format="%ad" -- $file_name)
  last_update_year=${last_update_year:0:4}
  copyright_year=$(sed '4q;d' ${file_name})
  copyright_year=${copyright_year#*-}
  copyright_year=${copyright_year:0:4}
  if [ "${last_update_year}" != "${copyright_year}" ] && [ "${copyright_year:0:2}" == "20" ]; then 
    echo "update ${copyright_year} to ${last_update_year} in ${file_name}${spaces}"
    sed -i '4s/'"-${copyright_year}"'/'"-${last_update_year}"'/g' ${file_name}
  else
    printf "scan ${directory}/${file_name}${spaces}\r"
  fi
done
cd ~-
}

UpdateCopyrights ../../src/Simd
UpdateCopyrights ../../src/Test
UpdateCopyrights ../../src/Use