#!/usr/bin/perl

# convert potential parameter file from old GalPot format to the new INI format
if($#ARGV<0) { die "Provide the file name to convert"; }
$FN = $ARGV[0];
open FI, "<$FN" or die "Can't open input file $FN";
$FN =~ s/\.t?pot/.ini/i;
open FO, ">$FN" or die "Can't open output file $FN";
print FO "#Units: 1 Msun, 1 Kpc\n\n";

$_=<FI>;  # number of disks
$ND = $_*1;
for($i=0; $i<$ND; $i++) {
    $_=<FI>;
    if(/([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)/i) {
        print FO "[Potential$ind]\n".
        "type = Disk\n".
        "surfaceDensity = $1\n".
        "scaleRadius = $2\n".
        "scaleHeight = $3\n".
        "innerCutoffRadius = $4\n".
        "modulationAmplitude = $5\n\n";
        $ind++;
    } else { die "Unknown format: $_"; }
}

$_=<FI>;  # number of spheroids
$NS = $_*1;
for($i=0; $i<$NS; $i++) {
    $_=<FI>;
    if(/([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)\s+([\d\+\-\.e]+)/i) {
        print FO "[Potential$ind]\n".
        "type = Spheroid\n".
        "densityNorm = $1\n".
        "axisRatioZ = $2\n".
        "gamma = $3\n".
        "beta = $4\n".
        "scaleRadius = $5\n".
        "outerCutoffRadius = $6\n\n";
        $ind++;
    } else { die "Unknown format: $_"; }
}
close FI;
close FO;
print "Created file $FN\n";
