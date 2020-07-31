#!/usr/bin/perl

# convert potential expansion coefficients from the old format (before July 2020)
# to the new format, which is unified with the INI format for all potential params
if($#ARGV<0) { die "Provide the file name to convert"; }
$FN = $ARGV[0];
open FI, "<$FN" or die "Can't open input file $FN";
$FN .= ".ini";
open FO, ">$FN" or die "Can't open output file $FN";
$_=<FI>;
chomp;
$mul = $_ eq "Multipole";
$cyl = $_ eq "CylSpline";
if(!$mul and !$cyl) { die "Invalid input file format"; }
print FO "[Potential]\ntype=$_\n";
$_=<FI>;
if($mul) {
  s/\t#n_radial//;
  print FO "gridSizeR=$_";
  $_=<FI>;
  s/\t#l_max//;
  print FO "lmax=$_";
  <FI>;
} elsif($cyl) {
  s/\t#size_R//;
  print FO "gridSizeR=$_";
  $_=<FI>;
  s/\t#size_z//;
  print FO "gridSizeZ=$_";
  $_=<FI>;
  s/\t#m_max//;
  print FO "mmax=$_";
}
print FO "Coefficients\n";
while(<FI>) { print FO; }
close FI;
close FO;
print "Created file $FN\n";
