
using System;
using System.IO;
using System.Runtime.InteropServices;


class Program
{
		
		
		public static void Main(string[] args)
		{
			RealSR.test(args);
		}
}



public class NCNNbase
{
					
		
	public static uint FloorLog2(uint x)
	{
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);

		return (uint)(NumBitsSet(x) - 1);
	}

	public static uint CeilingLog2(uint x)
	{
		int y = (int)(x & (x - 1));

		y |= -y;
		y >>= (WORDBITS - 1);
		x |= (x >> 1);
		x |= (x >> 2);
		x |= (x >> 4);
		x |= (x >> 8);
		x |= (x >> 16);

		return (uint)(NumBitsSet(x) - 1 - y);
	}

	public static int NumBitsSet(uint x)
	{
		x -= ((x >> 1) & 0x55555555);
		x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
		x = (((x >> 4) + x) & 0x0f0f0f0f);
		x += (x >> 8);
		x += (x >> 16);

		return (int)(x & 0x0000003f);
	}

	const int WORDBITS = 32;
		
	[UnmanagedFunctionPointer(CallingConvention.StdCall)]
		public delegate int FindFileFunc(IntPtr inpath, IntPtr outpath, IntPtr fmt, IntPtr vlist);
		
	[UnmanagedFunctionPointer(CallingConvention.StdCall)]
		public unsafe delegate void EncodeFunc(int IOid, int width, int height, int channel, int is16bit, byte* data);
		
	[UnmanagedFunctionPointer(CallingConvention.StdCall)]
		public unsafe delegate byte* DecodeFunc(int IOid, ref int width, ref int height, ref int channel, ref int is16bit, ref int ScaleParamLen, ref ScaleParam[] newparam);
		
		
		
	 	
	[Serializable]
	[StructLayout(LayoutKind.Explicit, Size = 8)]
	public struct ScaleParam
	{
		[FieldOffset(0)] public float DstScale;
		[FieldOffset(0)] public ushort DstW;
		[FieldOffset(2)] public ushort DstH;
		[FieldOffset(4)] public byte model;
		[FieldOffset(5)] public byte x1;
		[FieldOffset(6)] public byte x2;
		[FieldOffset(7)] public byte x3;
			
	}
		
	public static string intztdir = @"E:\r\cud\ncnn\src\bin\Release\tiztk";
	public static string outtztdir = @"E:\r\cud\ncnn\src\bin\Release\tztouty";
		
	public static string[] InNames;
	public static string[] OutNames;
	public static ScaleParam[][] sparams;
	public static double basicUpSample = 2;
	public static void testsklpa()
	{
		kjj:
		string ff = Console.ReadLine();
			
		if (ff[0] == 'x')
			return;
			
		string[] ffp = ff.Split(',');
			
		ScaleParam[] syk;
		if (ffp.Length == 2) {
			syk = makesparam(166, 250, dstw: int.Parse(ffp[0]), dsth: int.Parse(ffp[1]));
		} else {
			syk = makesparam(0, 0, scale: double.Parse(ffp[0]));
		}
			
		int steps = syk.Length;
			
		Console.WriteLine(steps + " steps");
		for (int i = 0; i < steps; i++) {
			Console.WriteLine(syk[i].DstScale + "\t//" + (2.0f * syk[i].DstScale));
		}
		goto kjj;
	}
	
	public unsafe static void dumpScaleParam(ScaleParam[] src)
	{
		int otbbl=src.Length;
		byte[] otb = new byte[otbbl*8];
		fixed(byte* srcbb=&otb[0])
		{
			for(int i=0;i<otbbl;i++)
			{
				*(ScaleParam*)(srcbb+8*i)=src[i];
			}
		}
		
		File.WriteAllBytes("pdum.bin",otb);
	}
	
		
	public static ScaleParam[] makesparam(int origw, int origh, double scale = 0, int dstw = 0, int dsth = 0)
	{
		ScaleParam[] k;
		double dstscale;
			
		bool usedstwh = false;
			
		if (scale == 0) {
			double dstsklw = 144514;
			double dstsklh = 144514;
				
			
			if (dstw != 0) {
				usedstwh = true;
				dstsklw = ((double)dstw) / ((double)origw);
			}
			if (dsth != 0) {
					
				usedstwh = true;
				dstsklh = ((double)dsth) / ((double)origh);
			}
				
			dstscale = Math.Min(dstsklw, dstsklh);
				
				
				
				
		} else {
				
			dstscale = scale;
		}
			
		if (dstscale <= basicUpSample)
			goto retdummy;
			
			
		int steps = (int)Math.Ceiling(Math.Log(dstscale, basicUpSample));
		k = new ScaleParam[steps];
			
		if (steps == 2) {
			k[0].DstScale = (float)(dstscale / (basicUpSample * basicUpSample));
			k[0].model = 0;
						
			goto retmulti;
		}
			
			
		int skll = ((int)basicUpSample) / 2;
		double upl = (double)((int)1 << (steps * skll));
			
			
			
		if (dstscale == upl) {
				
			k[0].DstScale = 0.0f;
			k[0].model = 0;
			for (int i = 1; i < steps; i++) {
				k[i].DstScale = 0.0f;
				k[i].model = 1;
			}
				
				
			goto retmulti;
		}
			
			
			
			
		double downer = Math.Pow(dstscale / upl, 1.0 / (double)((steps - 1) * (3 * steps - 2) / 2)); //(steps+2)/2)); //(steps)/2));
		double maxdowner = Math.Pow(downer, 2 * (steps - 1));//,steps); //,steps-1);
			
		k[0].DstScale = (float)maxdowner;
		k[0].model = 0;
			
			
			
			
			
			
				
		for (int i = 1; i < steps - 1; i++) {
			maxdowner /= downer;

			k[i].DstScale = (float)maxdowner;//dyv;
			k[i].model = 1;
		}
				
				
		retmulti:
		{
			
			if (usedstwh) {
				k[steps - 1].DstW = (ushort)(((double)origw * dstscale / basicUpSample) + 0.5);
				k[steps - 1].DstH = (ushort)(((double)origh * dstscale / basicUpSample) + 0.5);
				
			} else {
				k[steps - 1].DstScale = 0.0f;
				
			}
			
			k[steps - 1].model = 1;
			
			
			return k;
		}
			
		retdummy:
		{
			k = new ScaleParam[1];
			k[0].DstScale = 0;
			k[0].model = 0;
			return k;
		}
			
			
			
	}
		
		
		
	
		
}

public class RealSR : NCNNbase
{
	
	[DllImport("RealSRLIB.dll", CharSet = CharSet.Unicode)]
	public static extern void InOutList(int count, string[] in_paths, string[] out_paths);
	[DllImport("RealSRLIB.dll", CharSet = CharSet.Unicode)]
	public static extern int runRealSR(int argc, string[] argv);
	[DllImport("RealSRLIB.dll", CharSet = CharSet.Unicode)]
	public unsafe static extern byte* wic_decode_image(string File, ref int w, ref int h, ref int c);

		
	[DllImport("RealSRLIB.dll")]
	public static extern void SetFindFileFunc(FindFileFunc func, bool setisDir);
		
	[DllImport("RealSRLIB.dll")]
	public static extern void SetEncodeFunc(EncodeFunc func);
		
	[DllImport("RealSRLIB.dll")]
	public static extern int SetDecodeFunc(DecodeFunc func);
		
		

	public static unsafe void test(string[] args)
	{
			
		//intztdir=args[0];
		//outtztdir=args[1];
		//.etc 
			
			
		SetFindFileFunc(findfunc_example, true);
		basicUpSample = (double)SetDecodeFunc(decfunc_example);
		//SetEncodeFunc(encfunc_example);
		
		
		string[] cmdlines = { string.Empty,
			"-i", intztdir,
			"-o", outtztdir };
		
		
		runRealSR(cmdlines.Length, cmdlines);
	}
		
	public static unsafe byte* decfunc_example(int IOid, ref int width, ref int height, ref int channel, ref int is16bit, ref int ScaleParamLen, ref ScaleParam[] newparam)
	{
		is16bit = 1;
		byte* rett = wic_decode_image(InNames[IOid], ref width, ref height, ref channel);
			
		
			
		var newparam0 = makesparam(width, height, dstw: 2880, dsth: 4320);
		//dumpScaleParam(newparam0);
		//Console.WriteLine("InC#: width=" + width + ", height=" + height);
		//Console.ReadKey();
		
		
		sparams[IOid] = newparam0;
		newparam = newparam0;
		ScaleParamLen = newparam0.Length;
		return rett;
	}
		
	public static unsafe void encfunc_example(int IOid, int width, int height, int channel, int is16bit, byte* data)
	{
			
		byte[] output = new byte[width * height * channel * is16bit];
		Marshal.Copy((IntPtr)data, output, 0, output.Length);
		File.WriteAllBytes(OutNames[IOid], output);
	}
		
	public static int findfunc_example(IntPtr inpath, IntPtr outpath, IntPtr fmt, IntPtr vlist)
	{
		string[] drr = Directory.GetFiles(intztdir, "*", SearchOption.TopDirectoryOnly);
		int simplen = drr.Length;
		string[] innams = new string[simplen];
		string[] outnams = new string[simplen];
			
		int realget = 0;
			
		for (int i = 0; i < simplen; i++) {
			string oona = drr[i].Replace(intztdir, outtztdir) + ".png";
			if (!File.Exists(oona)) {
				innams[realget] = drr[i];
				outnams[realget] = oona;
				realget++;
			}
		}
			
		if (realget != 0) {
			InNames = innams;
			OutNames = outnams;
			sparams = new ScaleParam[realget][];
				
			InOutList(realget, InNames, OutNames);
		}
			
		return realget;
			
			
	}
}

public class W2X : NCNNbase
{
	
	[DllImport("W2XLIB.dll", CharSet = CharSet.Unicode)]
	public static extern void InOutList(int count, string[] in_paths, string[] out_paths);
	[DllImport("W2XLIB.dll", CharSet = CharSet.Unicode)]
	public static extern int runW2X(int argc, string[] argv);
	[DllImport("W2XLIB.dll", CharSet = CharSet.Unicode)]
	public unsafe static extern byte* wic_decode_image(string File, ref int w, ref int h, ref int c);

		
	[DllImport("W2XLIB.dll")]
	public static extern void SetFindFileFunc(FindFileFunc func, bool setisDir);
		
	[DllImport("W2XLIB.dll")]
	public static extern void SetEncodeFunc(EncodeFunc func);
		
	[DllImport("W2XLIB.dll")]
	public static extern int SetDecodeFunc(DecodeFunc func);
		
		

	public static unsafe void test(string[] args)
	{
			
		//intztdir=args[0];
		//outtztdir=args[1];
		//.etc 
			
			
		SetFindFileFunc(findfunc_example, true);
		basicUpSample = (double)SetDecodeFunc(decfunc_example);
		//SetEncodeFunc(encfunc_example);
		
		
		string[] cmdlines = { string.Empty,
			"-i", intztdir,
			"-o", outtztdir };
		
		
		
		runW2X(cmdlines.Length, cmdlines);
	}
		
	public static unsafe byte* decfunc_example(int IOid, ref int width, ref int height, ref int channel, ref int is16bit, ref int ScaleParamLen, ref ScaleParam[] newparam)
	{
		is16bit = 1;
		byte* rett = wic_decode_image(InNames[IOid], ref width, ref height, ref channel);
			
		//Console.WriteLine("InC#: width=" + width + ", height=" + height);
		//Console.ReadKey();
			
		var newparam0 = makesparam(width, height, dstw: 1280, dsth: 1920);
		sparams[IOid] = newparam0;
		newparam = newparam0;
		ScaleParamLen = newparam0.Length;
		return rett;
	}
		
	public static unsafe void encfunc_example(int IOid, int width, int height, int channel, int is16bit, byte* data)
	{
			
		byte[] output = new byte[width * height * channel * is16bit];
		Marshal.Copy((IntPtr)data, output, 0, output.Length);
		File.WriteAllBytes(OutNames[IOid], output);
	}
		
	public static int findfunc_example(IntPtr inpath, IntPtr outpath, IntPtr fmt, IntPtr vlist)
	{
		string[] drr = Directory.GetFiles(intztdir, "*", SearchOption.TopDirectoryOnly);
		int simplen = drr.Length;
		string[] innams = new string[simplen];
		string[] outnams = new string[simplen];
			
		int realget = 0;
			
		for (int i = 0; i < simplen; i++) {
			string oona = drr[i].Replace(intztdir, outtztdir) + ".png";
			if (!File.Exists(oona)) {
				innams[realget] = drr[i];
				outnams[realget] = oona;
				realget++;
			}
		}
			
		if (realget != 0) {
			InNames = innams;
			OutNames = outnams;
			sparams = new ScaleParam[realget][];
				
			InOutList(realget, InNames, OutNames);
		}
			
		return realget;
			
			
	}
}