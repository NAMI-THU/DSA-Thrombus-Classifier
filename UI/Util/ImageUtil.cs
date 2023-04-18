using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using itk.simple;
using Color = System.Drawing.Color;
using PixelFormat = System.Drawing.Imaging.PixelFormat;
using PixelId = itk.simple.PixelIDValueEnum;

namespace UI.Util;

/**
 * Some really black magic is happening here. We need this sorcery to correctly cast between multiple memory buffers and finally return a bitmap
 */
public static class ImageUtil
{
    [DllImport("gdi32.dll", EntryPoint = "DeleteObject")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool DeleteObject([In] IntPtr hObject);

    public static async Task<ImageSource> LoadNiftiToImage(string path)
    {
        var result = await Task.Run(() => {
            var input = SimpleITK.ReadImage(path);
            input = SimpleITK.Cast(input, PixelId.sitkUInt8);

            var buffer = input.GetBufferAsUInt8();

            var newBitmap = new Bitmap((int)input.GetWidth(), (int)input.GetHeight(), (int)input.GetWidth(),
                PixelFormat.Format8bppIndexed, buffer);
            var pal = newBitmap.Palette;
            for (var i = 0; i <= 255; i++) {
                // create greyscale color table
                pal.Entries[i] = Color.FromArgb(i, i, i);
            }

            newBitmap.Palette = pal; // you need to re-set this property to force the new ColorPalette

            var handle = newBitmap.GetHbitmap();
            try {
                var img = Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                img.Freeze();
                return img;
            } // We need to this manually to avoid a memory leak
            finally {
                DeleteObject(handle);
            }
        });
        return result;
    }

    public static async Task<ImageSource> Array2Image(float[][] image)
    {
        var result = await Task.Run(() => {
            var width = image.Length;
            var height = image[0].Length;
            var newBitmap = new Bitmap(width, height);

            for (var j = 0; j < width; j++) {
                for (var i = 0; i < height; i++) {
                    var c = (int)image[i][j];
                    var color = Color.FromArgb(c, c, c);
                    newBitmap.SetPixel(j, height - i - 1, color);
                }
            }

            var handle = newBitmap.GetHbitmap();
            try {
                var img = Imaging.CreateBitmapSourceFromHBitmap(handle, IntPtr.Zero, Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                img.Freeze();
                return img;
            } // We need to this manually to avoid a memory leak
            finally {
                DeleteObject(handle);
            }
        });
        return result;
    }
}