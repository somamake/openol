#ifndef OPENOL_H
#define OPENOL_H
#include <fftw3.h>
#include <memory>
#include <complex>
#include <stdio.h>
// #include "Diffraction.h"
#include "olclass.h"
#include "olutils.h"
#include "olprop.h"
#include "olcgh.h"
#include "olobject.h"
#include "Diffuser.h"
#include "Lensarry.h"


#ifdef __NVCC__
#include "golclass.h"
#include "golutils.h"
#include "golprop.h"
#include "golcgh.h"
#endif
#endif