TRUNK_DIR="../.."
SIMD_VERSION_H="$TRUNK_DIR/src/Simd/SimdVersion.h"
SIMD_VERSION_H_TXT="$TRUNK_DIR/prj/txt/SimdVersion.h.txt"
USER_VERSION_TXT="$TRUNK_DIR/prj/txt/Version.txt"
VERSION_TXT="$TRUNK_DIR/prj/gcc/Version.txt"

if [ -e "$VERSION_TXT" ]
then
	LAST_VERSION=`cat $VERSION_TXT`
else
	LAST_VERSION="0"
fi

cp $USER_VERSION_TXT $VERSION_TXT
printf . >>$VERSION_TXT
svn info $TRUNK_DIR | grep Revision: | cut -c11->>$VERSION_TXT
VERSION=`cat $VERSION_TXT`

NEED_TO_UPDATE="0"
if [ "$LAST_VERSION" = "$VERSION" ] 
then
	echo "Last project version '$LAST_VERSION' is equal to current version '$VERSION'."
else
	echo "Last project version '$LAST_VERSION' is not equal to current version '$VERSION'."
	NEED_TO_UPDATE="1"
fi

if [ -e "$SIMD_VERSION_H" ] 
then
	echo "File '$SIMD_VERSION_H' is already exist."
else
	echo "File '$SIMD_VERSION_H' is not exist."
	NEED_TO_UPDATE="1"
fi

if [ "$NEED_TO_UPDATE" = "0" ] 
then
	echo "Skip updating of '$SIMD_VERSION_H' file."
else
	echo "Create or update '$SIMD_VERSION_H' file."
	cp $SIMD_VERSION_H_TXT $SIMD_VERSION_H
	sed "-i" "s/@VERSION@/$VERSION/g" $SIMD_VERSION_H
fi

