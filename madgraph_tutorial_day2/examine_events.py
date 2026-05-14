#!/usr/bin/env python3
# run with 
# python3 plot_born.py run_01/unweighted_events.lhe --plot


import gzip
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse

mg_event_dir = "MG5_aMC_v3_7_0/bin/"
# -------------------------------------------------
# 4-vector
# -------------------------------------------------
class FourVec:
    def __init__(self, px, py, pz, E):
        self.px = px
        self.py = py
        self.pz = pz
        self.E = E

    def __add__(self, o):
        return FourVec(
            self.px + o.px,
            self.py + o.py,
            self.pz + o.pz,
            self.E + o.E
        )

    def pt(self):
        return math.sqrt(self.px**2 + self.py**2)

    def mass(self):
        m2 = self.E**2 - self.px**2 - self.py**2 - self.pz**2
        return math.sqrt(max(m2, 0.0))

    def eta(self):
        return 0.5 * math.log((self.E + self.pz) / (self.E - self.pz))

def smart_open(fname):
    with open(fname, "rb") as fb:
        magic = fb.read(2)

    if magic == b"\x1f\x8b":  # gzip magic number
        return gzip.open(fname, "rt")
    else:
        return open(fname, "rt")

# -------------------------------------------------
# Read LHE file
# -------------------------------------------------
def read_lhe(fname, max_nevents = -1, verbose=False):
    mll = []
    ptll = []
    etall = []
    with smart_open(fname) as f:
        in_event = False
        electrons = []
        for line in f:
          if "<event>" in line:
              in_event = True
              electrons = []
              continue
          if "</event>" in line and in_event:
              if len(electrons) == 2:
                  ll = electrons[0] + electrons[1]
                  mll.append(ll.mass())
                  ptll.append(ll.pt())
                  etall.append(ll.eta())
                  if verbose:
                      print(f"Event {len(mll)}: mll={ll.mass():.3f} GeV, ptll={ll.pt():.3f} GeV, etall={ll.eta():.3f}")
              if(len(mll) == max_nevents):
                  break
              in_event = False
              continue
          if not in_event:
              continue

          # skip comments inside event
          if line.startswith("#"):
              continue

          parts = line.split()
          if len(parts) < 10:
              continue

          # LHE format:
          # id status px py pz E ...
          pdgid = int(parts[0])
          status = int(parts[1])
          if abs(pdgid) == 11 and status == 1:
              px = float(parts[6])
              py = float(parts[7])
              pz = float(parts[8])
              E  = float(parts[9])
              electrons.append(FourVec(px, py, pz, E))
             
    
    return np.array(mll), np.array(ptll), np.array(etall)

# -------------------------------------------------
# Read one HepMC file
# -------------------------------------------------
def read_hepmc(filename, max_nevents = -1, verbose=False):
    mll = []
    ptll = []
    etall = []

    with gzip.open(filename, "rt") as f:

        electrons = []

        for line in f:

            if line.startswith("E "):
                electrons = []

            elif line.startswith("P "):
                cols = line.split()

                pdgid  = int(cols[2])
                px     = float(cols[3])
                py     = float(cols[4])
                pz     = float(cols[5])
                E      = float(cols[6])
                status = int(cols[8])

                # status == 1 means final state
                if status == 1 and abs(pdgid) == 11:
                    electrons.append(FourVec(px, py, pz, E))

                if len(electrons) == 2:
                    ll = electrons[0] + electrons[1]
                    mll.append(ll.mass())
                    ptll.append(ll.pt())
                    etall.append(ll.eta())
                    electrons = []
                    if verbose:
                      print(f"Showered event {len(mll)}: mll={ll.mass():.3f} GeV, ptll={ll.pt():.3f} GeV, etall={ll.eta():.3f}")
                    if(len(mll) == max_nevents):
                      break
              
    return np.array(mll), np.array(ptll), np.array(etall)

# -------------------------------------------------
# Plot helper
# -------------------------------------------------
def plot(results, idx, xlabel, title, out, xlim):

    plt.figure(figsize=(7,5))

    for label in results:
        arr = results[label][idx]

        plt.hist(
            arr,
            bins=25,
            range=xlim,
            histtype="step",
            density=True,
            linewidth=1.8,
            label=label
        )

    plt.xlabel(xlabel)
    plt.ylabel("Normalized events")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    
def main():
    parser = argparse.ArgumentParser(description="Read LHE file")
    parser.add_argument(
        "filenames",
        nargs="+",   # one or more filenames
        help="Input .lhe or .lhe.gz files"
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=-1,
        help="Number of events to process per file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print details about each event"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, create plots for all distributions"
    )

    args = parser.parse_args()

    # gather results
    results = {}
    for fname in args.filenames:
        print(f"Processing file: {fname}")
        if "lhe" in fname: 
            mll, ptll, etall = read_lhe(mg_event_dir+fname, args.nevents, args.verbose)
        elif "hepmc" in fname:
            mll, ptll, etall = read_hepmc(mg_event_dir+fname, args.nevents, args.verbose)
        else:
            print(f"Unknown file type for {fname}, skipping.")
            continue
        results[fname] = (mll, ptll, etall)
    
    if args.plot:
        print(f"Plotting results...")
        # number = idx of the results array 
        plot(results, 0, r"$m_{\ell\ell}$ [GeV]", "Dilepton invariant mass", "mll.png", (20,130))
        plot(results, 1, r"$p_{T,\ell\ell}$ [GeV]", "Dilepton transverse momentum", "ptll.png", (0,100))
        plot(results, 2, r"$\eta_{\ell\ell}$", "Dilepton rapidity", "etall.png", (-5,5))
        plt.show()


# -------------------------------------------------
# Run
# -------------------------------------------------
main()

