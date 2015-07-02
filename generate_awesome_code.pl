
@car=("Car","x","y","z");
@cyl=("Cyl","R","z","phi");
@sph=("Sph","r","theta","phi");
@prolsph=("ProlSph","lambda","nu","phi");

sub coeftype {   # return the type of transformation coefficient: 0, 1, or something else (marked as -1)
    my ($asrc,$adst,$s,$d)=@_;
    my @src=(@$asrc[0], @$asrc[1], @$asrc[2]);
    my @dst=(@$adst[0], @$adst[1], @$adst[2]);
    if($s eq $d) { return 1; }
    if($s ~~ @dst || $d ~~ @src || ($s eq "z" && $d eq "phi" || $s eq "phi" && $d eq "z") ) { return 0; }
    return -1;
}

sub genstruct {
    my ($asrc,$adst) = @_;
    my @src=(@$asrc[1], @$asrc[2], @$asrc[3]);
    my @dst=(@$adst[1], @$adst[2], @$adst[3]);
    my $src=@$asrc[0], $dst=@$adst[0];
    # first derivatives
    print "    template<> struct PosDerivT<$src, $dst> {\n        double ";
    my $line="";
    foreach my $d(@dst) {
        foreach my $s(@src) {
            if(coeftype(\@src, \@dst, $s, $d) == -1) { $line.="d${d}d${s}, "; }
        }
    }
    $line=~s/\,\ $//;
    print "$line;\n    };\n";
    # second derivatives
    print "    template<> struct PosDeriv2T<$src, $dst> {\n        double ";
    my $line="";
    foreach my $d(@dst) {
        for (my $is1=0; $is1<3; $is1++) {
            my $s1=$src[$is1];
            for(my $is2=$is1; $is2<3; $is2++) {
                my $s2=$src[$is2];
                if($is1==$is2) {
                    if(coeftype(\@src, \@dst, $s1, $d) == -1) { $line.="d2${d}d${s1}2, "; }
                } else {
                    if(coeftype(\@src, \@dst, $s1, $d) == -1 && coeftype(\@src, \@dst, $s2, $d) == -1 ) 
                    { $line.="d2${d}d${s1}d${s2}, "; }
                }
            }
        }
    }
    $line=~s/\,\ $//;
    print "$line;\n    };\n\n";
}

sub gengrad {
    my ($asrc,$adst) = @_;
    my @src=(@$asrc[1], @$asrc[2], @$asrc[3]);
    my @dst=(@$adst[1], @$adst[2], @$adst[3]);
    my $src=@$asrc[0], $dst=@$adst[0];
    print "    template<>\n    Grad$dst toGrad(const Grad${src}& src, const PosDerivT<$dst, $src>& deriv) {\n        Grad$dst dest;\n";
    foreach my $d(@dst) { 
        print "        dest.d$d = ";
        my $line="";
        foreach my $s(@src) {
            my $coeftype=coeftype(\@src, \@dst, $s, $d);
            if($coeftype==1) { $line.="src.d$s + "; }
            elsif($coeftype==0) {}
            else { $line.="src.d$s*deriv.d${s}d${d} + "; }
        }
        $line =~ s/\ \+\ $//;
        print "$line;\n";
    }
    print "        return dest;\n    }\n\n";
}

sub genhess {
    my ($asrc,$adst) = @_;
    my @src=(@$asrc[1], @$asrc[2], @$asrc[3]);
    my @dst=(@$adst[1], @$adst[2], @$adst[3]);
    my $src=@$asrc[0], $dst=@$adst[0];
    print "    template<>\n    Hess$dst toHess(const Grad${src}& srcGrad, const Hess${src}& srcHess,\n".
    "        const PosDerivT<$dst, $src>& deriv, const PosDeriv2T<$dst, $src>& deriv2) {\n".
    "        Hess$dst dest;\n";
    for(my $id1=0; $id1<3; $id1++) { 
        for(my $id2=$id1; $id2<3; $id2++) {
            my $d1=$dst[$id1]; my $d2=$dst[$id2];
            my $suffixd=($id1==$id2)?"d${d1}2":"d${d1}d${d2}";
            my @line=();
            # matrix multiplication part
            for(my $is1=0; $is1<3; $is1++) { 
                my $s1=$src[$is1];
                my $coeftype1=coeftype(\@src, \@dst, $s1, $d1);
                my @line1=();
                for(my $is2=0; $is2<3; $is2++) {
                    my $s2=$src[$is2];
                    my $srcHess="srcHess." .
                        (($is1==$is2)?"d${s1}2":($is1<$is2)?"d${s1}d${s2}":"d${s2}d${s1}");  # component of the source hessian
                    my $coeftype2=coeftype(\@src, \@dst, $s2, $d2);
                    if($coeftype2!=0) {
                        push @line1, "$srcHess".($coeftype2==1?"":"*deriv.d${s2}d${d2}");
                    }
                }
                if($#line1>=0 && $coeftype1!=0) {
                    my $line1= ($#line1==0? $line1[0] : "(".join(" + ", @line1).")").
                        ($coeftype1==1?"":"*deriv.d${s1}d${d1}");
                    if(length($line1)>20) { $line1="\n            $line1"; }
                    push @line, $line1;
                }
            }
            # connection part
            my @line1=();
            foreach $s(@src) {
                my $coef="0";
                if($id1==$id2) {
                    if(coeftype(\@dst, \@src, $d1, $s) == -1) { $coef="d2${s}d${d1}2"; }
                } else {
                    if(coeftype(\@dst, \@src, $d1, $s) == -1 && coeftype(\@dst, \@src, $d2, $s) == -1 ) 
                    { $coef="d2${s}d${d1}d${d2}"; }
                }
                if($coef ne "0") {
                    push @line1, "srcGrad.d$s*deriv2.$coef";
                }
            }
            if($#line1>=0) { push @line, "\n            ".join(" + ", @line1); }
            if($#line>=0) {
                print "        dest.$suffixd = ".join(" + ", @line).";\n";
            }
        }
    }
    print "        return dest;\n    }\n\n";
}

genstruct(\@cyl, \@prolsph);
gengrad(\@prolsph, \@cyl);
genhess(\@prolsph, \@cyl);

if(0) {
gengrad(\@cyl, \@car);
gengrad(\@sph, \@car);
gengrad(\@car, \@cyl);
gengrad(\@sph, \@cyl);
gengrad(\@car, \@sph);
gengrad(\@cyl, \@sph);
}
if(0) {
genhess(\@cyl, \@car);
genhess(\@sph, \@car);
genhess(\@car, \@cyl);
genhess(\@sph, \@cyl);
genhess(\@car, \@sph);
genhess(\@cyl, \@sph);
}