using System.Windows.Media;
using System.Windows.Media.Imaging;
using Nifti.NET;

namespace ThromboMapUI.Util;

public class Utils
{
    private BitmapSource LoadNiftiToBitmap(string path)
    {
        // TODO Not working
        
        var nifti = NiftiFile.Read(path);
        var array = new short[nifti.Dimensions[0]][];
        for (var x = 0; x < nifti.Dimensions[0]; x++)
        {
            array[x] = new short[nifti.Dimensions[1]];
            for (var y = 0; y < nifti.Dimensions[1]; y++)
            {
                array[x][y] = nifti[x, y,0];
            }
        }
        
        var bitmap = BitmapSource.Create(nifti.Dimensions[0], nifti.Dimensions[1],1,1,PixelFormats.Default, BitmapPalettes.Gray256, array, 1);
        return bitmap;
    }
}