#################
#    HLS4ML
#################
array set opt {
  reset      0
  csim       0
  synth      1
  cosim      0
  validation 0
  vhdl       1
  verilog    1
  export     0
  vsynth     0
  bitfile    0
  fifo_opt   0
  ran_frame  2
  sw_opt     0
  power      0
  da         0
  bup        0
  fifo_depth 1
  fifo_depth_bypass 0
}

# Get pathname to this script to use as dereference path for relative file pathnames
set sfd [file dirname [info script]]

if { [info exists ::argv] } {
  foreach arg $::argv {
    foreach {optname optval} [split $arg '='] {}
    if { [info exists opt($optname)] } {
      if {[string is integer -strict $optval]} {
        set opt($optname) $optval
      } else {
        set opt($optname) [string is true -strict $optval]
      }
    }
  }
}

# Reconvergent-bypass (*_cpy*) channels need a deeper FIFO than regular (*_out)
# interconnect; fifo_depth_bypass sizes only the bypass class. A sentinel 0 means
# "not set" -> fall back to fifo_depth so existing single-knob builds are unchanged.
if { $opt(fifo_depth_bypass) == 0 } { set opt(fifo_depth_bypass) $opt(fifo_depth) }

puts "***** INVOKE OPTIONS *****"
foreach x [lsort [array names opt]] {
  puts "[format {   %-20s %s} $x $opt($x)]"
}
puts ""

proc report_time { op_name time_start time_end } {
  set time_taken [expr $time_end - $time_start]
  set time_s [expr ($time_taken / 1000) % 60]
  set time_m [expr ($time_taken / (1000*60)) % 60]
  set time_h [expr ($time_taken / (1000*60*60)) % 24]
  puts "***** ${op_name} COMPLETED IN ${time_h}h${time_m}m${time_s}s *****"
}

proc setup_xilinx_part { part } {
  # Map Xilinx PART into Catapult library names
  set part_sav $part
  set libname [lindex [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret v] 0]
  puts "Library Name: $libname"
  if { [llength $libname] == 1 } {
    set libpath [library get /CONFIG/PARAMETERS/Vivado/PARAMETERS/Xilinx/PARAMETERS/*/PARAMETERS/*/PARAMETERS/$part/LIBRARIES/*/NAME -match glob -ret p]
    puts "Library Path: $libpath"
    if { [regexp {/CONFIG/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/PARAMETERS/(\S+)/.*} $libpath dummy rtltool vendor family speed part] } {
      solution library add $libname -- -rtlsyntool $rtltool -vendor $vendor -family $family -speed $speed -part $part_sav
    } else {
      solution library add $libname -- -rtlsyntool Vivado
    }
  } else {
    logfile message "Could not find specific Xilinx base library for part '$part'. Using KINTEX-u\n" warning
    solution library add mgc_Xilinx-KINTEX-u-2_beh -- -rtlsyntool Vivado -manufacturer Xilinx -family KINTEX-u -speed -2 -part xcku115-flvb2104-2-i
  }
  solution library add Xilinx_RAMS
  solution library add Xilinx_ROMS
  solution library add Xilinx_FIFO
}


proc setup_altera_lib { } {
  # Default Catapult target: behavioral Agilex family library, queried the same way
  # setup_xilinx_part looks up Vivado parts (rtlsyntool/vendor/family/speed/part come
  # from Catapult's own /CONFIG/PARAMETERS/Quartus/PARAMETERS/Altera tree, not guessed).
  solution library add mgc_Altera-Agilex-2_beh -- -rtlsyntool Quartus -manufacturer Altera -family Agilex -speed 2 -part AGFB014R24B2E2V
  # Only the behavioral Altera memory libraries (a plain register array plus a ramstyle
  # attribute). Altera_FIFO and Altera_LPM are left out: they instantiate Intel IP
  # (scfifo/dcfifo/altera_syncram), which keeps the RTL from being vendor-neutral.
  # Catapult's generic ccs_sample_mem/ccs_sample_rom cannot be used here: Catapult rejects
  # them next to any FPGA family library (LIB-223), they only load in ASIC solutions.
  solution library add Altera_M20K
  solution library add Altera_MLAB
  solution library add Altera_DIST
  solution library add Altera_ROMS
}


proc find_array_resources { design pat } {
  # Resources matching pat up to four levels below the design (stage instance, nested block
  # instance such as im2col, core); a glob '*' does not cross '/'.
  set found {}
  foreach prefix [list $design $design/* $design/*/* $design/*/*/*] {
    foreach m2m [directive get -match glob -checkpath 0 -ret p $prefix/$pat/MAP_TO_MODULE] {
      lappend found [join [lrange [split $m2m /] 0 end-1] /]
    }
  }
  return [lsort -unique $found]
}


proc map_partitioned_arrays_to_registers { design } {
  # Arrays the Vivado templates partition completely (ARRAY_PARTITION complete), mapped to
  # registers so Catapult never builds them as a memory, whatever MEM_MAP_THRESHOLD is. Matched
  # by resource name (declaring function + array): hls_resource pragmas on these arrays make
  # Catapult crash intermittently once inlined into the conv path, and line_buffer.Array is an
  # ap_shift_reg member that no pragma can name. Templates that are their own block (im2col) are
  # anchored on the block name instead. Add an entry per template once it is reviewed.
  set patterns {
    *dense_resource_rf_*:acc:rsc
    *dense_resource_rf_*:acc_part:rsc
    *dense_resource_rf_*:tmpmult:rsc
    *dense_resource_rf_*:mult:rsc
    *compute_output_buffer_?d<*:kernel_data:rsc
    *compute_output_buffer_?d<*:res_out:rsc
    *pointwise_mult_buffer<*:data:rsc
    *pointwise_mult_buffer<*:res:rsc
    *shift_line_buffer<*:shift_buffer:rsc
    *conv_?d_buffer_cl<*:line_buffer.Array:rsc
    *im2col_?d_gemm_rows<*>/core/kernel_data:rsc
    *im2col_?d_gemm_rows<*>/core/line_buffer:rsc
    *im2col_?d_gemm_rows<*>/core/*shift_buffer:rsc
    *einsum_dense<*:data:rsc
    *einsum_dense<*:inp_tpose:rsc
    *einsum_dense<*:out_buffer:rsc
    *einsum_stream_impl<*::run:data0:rsc
    *einsum_stream_impl<*::run:data1:rsc
    *einsum_stream_impl<*::run:tpose_i0:rsc
    *einsum_stream_impl<*::run:tpose_i1:rsc
    *einsum_stream_rows<*:lane_acc:rsc
    *einsum_resource<*:lane_acc:rsc
  }
  foreach pat $patterns {
    foreach rsc [find_array_resources $design $pat] {
      logfile message "directive set $rsc -MAP_TO_MODULE {\[Register\]}\n" info
      directive set $rsc -MAP_TO_MODULE {[Register]}
    }
  }
}


proc keep_stream_packets_out_of_memory { design } {
  # nnet::array stream packets stay a packed word, as Vitis packs structs in streams by default:
  # channels without an explicit mapping stay on the FIFO Catapult picks by default (it would
  # otherwise turn them into a shared memory), and every packet member array (the nnet::array
  # 'data' member, e.g. data_pack.data) is mapped to registers. Matched by resource name because
  # no source pragma can name a struct member.
  foreach m2m [directive get -match glob -checkpath 0 -ret p $design/*:cns/MAP_TO_MODULE] {
    if { [directive get $m2m] eq "" } {
      set rsc [join [lrange [split $m2m /] 0 end-1] /]
      logfile message "directive set $rsc -MAP_TO_MODULE ccs_ioport.ccs_pipe\n" info
      directive set $rsc -MAP_TO_MODULE ccs_ioport.ccs_pipe
    }
  }
  foreach rsc [find_array_resources $design *.data:rsc] {
    logfile message "directive set $rsc -MAP_TO_MODULE {\[Register\]}\n" info
    directive set $rsc -MAP_TO_MODULE {[Register]}
  }
}


proc setup_asic_libs { args } {
  set do_saed 0
  foreach lib $args {
    solution library add $lib -- -rtlsyntool DesignCompiler
    if { [lsearch -exact {saed32hvt_tt0p78v125c_beh saed32lvt_tt0p78v125c_beh saed32rvt_tt0p78v125c_beh} $lib] != -1 } {
      set do_saed 1
    }
  }
  solution library add ccs_sample_mem
  solution library add ccs_sample_rom
  solution library add hls4ml_lib
  go libraries

  # special exception for SAED32 for use in power estimation
  if { $do_saed } {
    # SAED32 selected - enable DC settings to access Liberty data for power estimation
    source [application get /SYSTEM/ENV_MGC_HOME]/pkgs/siflibs/saed/setup_saedlib.tcl
  }
}

options set Input/CppStandard {c++17}
options set Input/CompilerFlags -DRANDOM_FRAMES=$opt(ran_frame)
options set Input/SearchPath {$MGC_HOME/shared/include/nnet_utils} -append
options set ComponentLibs/SearchPath {$MGC_HOME/shared/pkgs/ccs_hls4ml} -append

if {$opt(reset) && [file exists CATAPULT_DIR.ccs]} {
  project load CATAPULT_DIR.ccs
  go new
} else {
  project new -name CATAPULT_DIR
}

#--------------------------------------------------------
# Configure Catapult Options
# downgrade HIER-10
options set Message/ErrorOverride HIER-10 -remove
solution options set Message/ErrorOverride HIER-10 -remove

if {$opt(vhdl)}    {
  options set Output/OutputVHDL true
} else {
  options set Output/OutputVHDL false
}
if {$opt(verilog)} {
  options set Output/OutputVerilog true
} else {
  options set Output/OutputVerilog false
}

#--------------------------------------------------------
# Configure Catapult Flows
if { [info exists ::env(XILINX_PCL_CACHE)] } {
options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
solution options set /Flows/Vivado/PCL_CACHE $::env(XILINX_PCL_CACHE)
}

# Turn on HLS4ML flow (wrapped in a cache so that older Catapult installs still work)
catch {flow package require /HLS4ML}

# Turn on SCVerify flow
flow package require /SCVerify
#  flow package option set /SCVerify/INVOKE_ARGS {$sfd/firmware/weights $sfd/tb_data/tb_input_features.dat $sfd/tb_data/tb_output_predictions.dat}
#hls-fpga-machine-learning insert invoke_args

# Turn on VSCode flow
# flow package require /VSCode
# To launch VSCode on the C++ HLS design:
#   cd my-Catapult-test
#   code Catapult.code-workspace

#--------------------------------------------------------
#    Start of HLS script
set design_top myproject
solution file add $sfd/firmware/myproject.cpp
solution file add $sfd/myproject_test.cpp -exclude true

#hls-fpga-machine-learning insert blackboxes

# Parse parameters.h to determine config info to control directives/pragmas
set IOType io_stream
if { ![file exists $sfd/firmware/parameters.h] } {
  logfile message "Could not locate firmware/parameters.h. Unable to determine network configuration.\n" warning
} else {
  set pf [open "$sfd/firmware/parameters.h" "r"]
  while {![eof $pf]} {
    gets $pf line
    if { [string match {*io_type = nnet::io_stream*} $line] } {
      set IOType io_stream
      break
    }
  }
  close $pf
}

if { $IOType == "io_stream" } {
solution options set Architectural/DefaultRegisterThreshold 2050
}
directive set -RESET_CLEARS_ALL_REGS no
# Constrain arrays to map to memory only over a certain size. Matches the Vivado flow's
# config_array_partition -complete_threshold 4096 (MaximumSize default), so unannotated
# arrays get the same register/memory policy in both backends.
directive set -MEM_MAP_THRESHOLD 4096
# The following line gets modified by the backend writer
set hls_clock_period 5

go analyze

# NORMAL TOP DOWN FLOW
if { ! $opt(bup) } {

go compile

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  set time_start [clock clicks -milliseconds]
  flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
  set time_end [clock clicks -milliseconds]
  report_time "C SIMULATION" $time_start $time_end
}

puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs

directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

# Optimize for latency rather than the default area goal. Area goal drives rshare to
# time-multiplex the fully-unrolled multiplier cone across many FSM states; latency goal
# keeps the spatial unroll the per-loop hls_unroll pragmas intend.
directive set -DESIGN_GOAL latency

# Loop parallelism (unroll, pipeline II=reuse_factor) is expressed in-source in nnet_utils with
# Catapult's own pragmas (#pragma hls_unroll / #pragma hls_pipeline_init_interval), the same
# way the Vivado/Vitis templates carry #pragma HLS UNROLL / PIPELINE. Catapult honors them, so no
# per-loop directives are set here. We deliberately do NOT use a global `-UNROLL yes`: it silently
# disconnects an output lane (stuck constant, fails cosim) and breaks timing.

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]

  go assembly
  set design [solution get -name]
  logfile message "Adjusting FIFO_DEPTH for top-level interconnect channels\n" warning
  # FIFO interconnect between layers
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_out:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo $opt(fifo_depth)\n" info
    directive set -match glob "$ch_fifo" $opt(fifo_depth)
  }
  # Bypass/reconvergent paths (e.g. clone fan-out feeding a late einsum operand) need depth
  # > 1 to avoid dataflow deadlock; honor the build-time fifo_depth (the in-source
  # #pragma hls_fifo_depth the writer emits is ignored by Catapult; only this directive is).
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_cpy*:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo $opt(fifo_depth_bypass) (bypass)\n" info
    directive set -match glob "$ch_fifo" $opt(fifo_depth_bypass)
  }
  # Per-boundary overrides (HLSConfig InputFifoDepth) win over the blanket loops above.
  #hls-fpga-machine-learning insert fifo-depth-overrides
  map_partitioned_arrays_to_registers $design
  keep_stream_packets_out_of_memory $design

  go architect

  go allocate

  go schedule

  go extract
  set time_end [clock clicks -milliseconds]
  report_time "C/RTL SYNTHESIS" $time_start $time_end
}

# BOTTOM-UP FLOW
} else {
  # Start at 'go analyze'
  go analyze

  # Build the design bottom-up
  directive set -CLOCKS [list clk [list -CLOCK_PERIOD $hls_clock_period -CLOCK_EDGE rising -CLOCK_OFFSET 0.000000 -CLOCK_UNCERTAINTY 0.0 -RESET_KIND sync -RESET_SYNC_NAME rst -RESET_SYNC_ACTIVE high -RESET_ASYNC_NAME arst_n -RESET_ASYNC_ACTIVE low -ENABLE_NAME {} -ENABLE_ACTIVE high]]

  set blocks [solution get /HIERCONFIG/USER_HBS/*/RESOLVED_NAME -match glob -rec 1 -ret v -state analyze]
  set bu_mappings {}
  set top [lindex $blocks 0]
  foreach block [lreverse [lrange $blocks 1 end]] {
    # skip blocks that are net nnet:: functions
    if { [string match {nnet::*} $block] == 0 } { continue }
    go analyze
    solution design set $block -top
    go compile
    solution library remove *
    puts "***** SETTING TECHNOLOGY LIBRARIES *****"
#hls-fpga-machine-learning insert techlibs
    go extract
    set block_soln "[solution get /TOP/name -checkpath 0].[solution get /VERSION -checkpath 0]"
    lappend bu_mappings [solution get /CAT_DIR] /$top/$block "\[Block\] $block_soln"
  }

  # Move to top design
  go analyze
  solution design set $top -top
  go compile

  if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    set time_start [clock clicks -milliseconds]
    flow run /SCVerify/launch_make ./scverify/Verify_orig_cxx_osci.mk {} SIMTOOL=osci sim
    set time_end [clock clicks -milliseconds]
    report_time "C SIMULATION" $time_start $time_end
  }
  foreach {d i l} $bu_mappings {
    logfile message "solution options set ComponentLibs/SearchPath $d -append\n" info
    solution options set ComponentLibs/SearchPath $d -append
  }

  # Add bottom-up blocks
  puts "***** SETTING TECHNOLOGY LIBRARIES *****"
  solution library remove *
#hls-fpga-machine-learning insert techlibs
  # need to revert back to go compile
  go compile
  foreach {d i l} $bu_mappings {
    logfile message "solution library add [list $l]\n" info
    eval solution library add [list $l]
  }
  go libraries

  # Map to bottom-up blocks
  foreach {d i l} $bu_mappings {
    # Make sure block exists
    set cnt [directive get $i/* -match glob -checkpath 0 -ret p]
    if { $cnt != {} } {
      logfile message "directive set $i -MAP_TO_MODULE [list $l]\n" info
      eval directive set $i -MAP_TO_MODULE [list $l]
    }
  }
  go assembly
  set design [solution get -name]
  logfile message "Adjusting FIFO_DEPTH for top-level interconnect channels\n" warning
  # FIFO interconnect between layers
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_out:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo $opt(fifo_depth)\n" info
    directive set -match glob "$ch_fifo" $opt(fifo_depth)
  }
  # Bypass/reconvergent paths (e.g. clone fan-out feeding a late einsum operand) need depth
  # > 1 to avoid dataflow deadlock; honor the build-time fifo_depth (the in-source
  # #pragma hls_fifo_depth the writer emits is ignored by Catapult; only this directive is).
  foreach ch_fifo_m2m [directive get -match glob -checkpath 0 -ret p $design/*_cpy*:cns/MAP_TO_MODULE] {
    set ch_fifo [join [lrange [split $ch_fifo_m2m '/'] 0 end-1] /]/FIFO_DEPTH
    logfile message "directive set -match glob $ch_fifo $opt(fifo_depth_bypass) (bypass)\n" info
    directive set -match glob "$ch_fifo" $opt(fifo_depth_bypass)
  }
  # Per-boundary overrides (HLSConfig InputFifoDepth) win over the blanket loops above.
  #hls-fpga-machine-learning insert fifo-depth-overrides
  map_partitioned_arrays_to_registers $design
  keep_stream_packets_out_of_memory $design
  go architect
  go allocate
  go schedule
  go dpfsm
  go extract
}

project save

if {$opt(cosim) || $opt(validation)} {
  if {$opt(verilog)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_v_msim.mk {} SIMTOOL=msim sim
  }
  if {$opt(vhdl)} {
    flow run /SCVerify/launch_make ./scverify/Verify_rtl_vhdl_msim.mk {} SIMTOOL=msim sim
  }
}

if {$opt(export)} {
  puts "***** EXPORT IP *****"
  set time_start [clock clicks -milliseconds]
# Not yet implemented. Do we need to include value of $version ?
#  flow package option set /Vivado/BoardPart xilinx.com:zcu102:part0:3.1
#  flow package option set /Vivado/IP_Taxonomy {/Catapult}
#  flow run /Vivado/launch_package_ip -shell ./vivado_concat_v/concat_v_package_ip.tcl
  set time_end [clock clicks -milliseconds]
  report_time "EXPORT IP" $time_start $time_end
}
if {$opt(sw_opt)} {
  puts "***** Pre Power Optimization *****"
  go switching
  if {$opt(verilog)} {
    flow run /PowerAnalysis/report_pre_pwropt_Verilog
  }
  if {$opt(vhdl)} {
    flow run /PowerAnalysis/report_pre_pwropt_VHDL
  }
}

if {$opt(power)} {
  puts "***** Power Optimization *****"
  go power
}

if {$opt(vsynth)} {
  puts "***** VIVADO SYNTHESIS *****"
  set time_start [clock clicks -milliseconds]
  flow run /Vivado/synthesize -shell vivado_concat_v/concat_rtl.v.xv
  set time_end [clock clicks -milliseconds]
  report_time "VIVADO SYNTHESIS" $time_start $time_end
}

if {$opt(bitfile)} {
  puts "***** Option bitfile not supported yet *****"
}

if {$opt(da)} {
  puts "***** Launching DA *****"
  flow run /DesignAnalyzer/launch
}

if { [catch {flow package present /HLS4ML}] == 0 } {
  flow run /HLS4ML/collect_reports
}
