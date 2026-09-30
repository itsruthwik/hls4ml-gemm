module rp_ram
#(
parameter data_width = 128,
parameter addr_width = 7,
parameter depth = 128
)(
	radr, wadr, d, we, re, clk, q
);
	input [addr_width-1:0] radr;
	input [addr_width-1:0] wadr;
	input [data_width-1:0] d;
	input we;
	input re;
	input clk;
	output reg [data_width-1:0] q;

	(* ramstyle = "M20K" *) reg [data_width-1:0] mem [depth-1:0];

	always @(posedge clk) begin
		if (we) mem[wadr] <= d;
		if (re) q <= mem[radr];
	end
endmodule
