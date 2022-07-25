using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using itk.simple;
using Nifti.NET;

namespace ThromboMapUI.Util;

public class ImageUtil
{
    public static async Task<BitmapSource> LoadNiftiToBitmap(string path)
    {
        // TODO Not working
        var result = await Task.Run(() => { 
            var nifti = NiftiFile.Read(path);
            var array = new float[nifti.Dimensions[0]][];
            for (var x = 0; x < nifti.Dimensions[0]; x++)
            {
                array[x] = new float[nifti.Dimensions[1]];
                for (var y = 0; y < nifti.Dimensions[1]; y++)
                {
                    array[x][y] = (float) nifti[x, y,0];
                }
            }
            
            var bitmap = BitmapSource.Create(nifti.Dimensions[0], nifti.Dimensions[1],1,1,PixelFormats.Gray32Float, BitmapPalettes.Gray256, array, 1);
            return bitmap;
        });
        return result;
    }
}