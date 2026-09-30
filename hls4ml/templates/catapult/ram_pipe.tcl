# RAM-backed channel pipe for wide, deep inter-block channels (Altera flow).
#
# ccs_pipe stores every FIFO word in registers, so a wide channel with a large depth (e.g. a
# residual bypass held for a whole attention block) costs width*depth flip-flops and its
# read mux. rp_ram is a behavioural 1R1W RAM (sync read, ramstyle M20K, no translate_off) that
# MemGen wraps into a pipe (GEN_RAM_PIPE): still one word per cycle at II=1 on both sides, but
# three cycles from the write to the first read. The generated pipe module is always named
# my_mem_pipe. The range limits below only bound the generics MemGen accepts; a channel outside
# them stays on ccs_pipe (see map_ram_pipes).
set ram_pipe_max_width 4096
set ram_pipe_max_depth 65536

proc build_ram_pipe_lib { sfd libdir } {
  global ram_pipe_max_width ram_pipe_max_depth
  file mkdir $libdir
  # MemGen records the model file path relative to its output dir, so stage the copy locally.
  file copy -force $sfd/rp_ram.v $libdir/rp_ram.v
  set addr_max [expr {int(ceil(log($ram_pipe_max_depth) / log(2)))}]
  flow package require MemGen
  # RTLTOOL must be Quartus: DesignCompiler is rejected next to the Agilex library (LIB-223).
  flow run /MemGen/MemoryGenerator_BuildLib [subst -nocommands {
    VENDOR           *
    RTLTOOL          Quartus
    TECHNOLOGY       *
    LIBRARY          rp_ram
    MODULE           rp_ram
    OUTPUT_DIR       $libdir/memgenout
    FILES {
      { FILENAME $libdir/rp_ram.v FILETYPE Verilog MODELTYPE generic PARSE 1 PATHTYPE copy STATICFILE 1 }
    }
    VHDLARRAYPATH    {}
    WRITEDELAY       0.1
    INITDELAY        1
    READDELAY        0.1
    VERILOGARRAYPATH {}
    GEN_RAM_PIPE     1
    INPUTDELAY       0.01
    TIMEUNIT         1ns
    WIDTH            data_width
    AREA             123
    RDWRRESOLUTION   RBW
    WRITELATENCY     1
    READLATENCY      1
    DEPTH            depth
    PARAMETERS {
      { PARAMETER data_width TYPE hdl IGNORE 0 MIN 2 MAX $ram_pipe_max_width DEFAULT 128 }
      { PARAMETER addr_width TYPE hdl IGNORE 0 MIN 1 MAX $addr_max DEFAULT 7 }
      { PARAMETER depth      TYPE hdl IGNORE 0 MIN 1 MAX $ram_pipe_max_depth DEFAULT 128 }
    }
    PORTS {
      { NAME port_0 MODE Read  }
      { NAME port_1 MODE Write }
    }
    PINMAPS {
      { PHYPIN radr LOGPIN ADDRESS      DIRECTION in  WIDTH addr_width PHASE {} DEFAULT {} PORTS port_0 }
      { PHYPIN wadr LOGPIN ADDRESS      DIRECTION in  WIDTH addr_width PHASE {} DEFAULT {} PORTS port_1 }
      { PHYPIN d    LOGPIN DATA_IN      DIRECTION in  WIDTH data_width PHASE {} DEFAULT {} PORTS port_1 }
      { PHYPIN we   LOGPIN WRITE_ENABLE DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_1 }
      { PHYPIN re   LOGPIN READ_ENABLE  DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS port_0 }
      { PHYPIN clk  LOGPIN CLOCK        DIRECTION in  WIDTH 1.0        PHASE 1  DEFAULT {} PORTS {port_0 port_1} }
      { PHYPIN q    LOGPIN DATA_OUT     DIRECTION out WIDTH data_width PHASE {} DEFAULT {} PORTS port_0 }
    }
  }]
  options set ComponentLibs/SearchPath $libdir/memgenout -append
}

# Map every internal channel that is wider than min_width bits and at least min_depth deep to
# the RAM pipe. Runs after the FIFO_DEPTH directives (the depth is a build option as well as a
# per-boundary override, so it is only known here) and after keep_stream_packets_out_of_memory.
# Only channels still on the default ccs_pipe are remapped: a channel mapped explicitly to
# anything else keeps its mapping.
proc map_ram_pipes { design widths min_width min_depth } {
  global ram_pipe_max_width ram_pipe_max_depth
  foreach m2m [directive get -match glob -checkpath 0 -ret p $design/*:cns/MAP_TO_MODULE] {
    set rsc [join [lrange [split $m2m /] 0 end-1] /]
    set name [regsub {:cns$} [lindex [split $rsc /] end] {}]
    if { ![dict exists $widths $name] } { continue }
    set width [dict get $widths $name]
    set cur [directive get $m2m]
    if { $cur ne "" && $cur ne "ccs_ioport.ccs_pipe" } { continue }
    set depth [directive get $rsc/FIFO_DEPTH]
    if { ![string is integer -strict $depth] } { continue }
    if { $width <= $min_width || $depth < $min_depth } { continue }
    if { $width > $ram_pipe_max_width || $depth > $ram_pipe_max_depth } {
      logfile message "$name ($width bits x $depth) is outside the RAM pipe range, kept on ccs_pipe\n" warning
      continue
    }
    logfile message "directive set $rsc -MAP_TO_MODULE rp_ram.my_mem_pipe ($width bits x $depth)\n" info
    directive set $rsc -MAP_TO_MODULE rp_ram.my_mem_pipe
  }
}
