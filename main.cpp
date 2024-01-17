#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <gtk/gtk.h>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <ostream>

using namespace std;
using namespace cv;
const int baseline = 58;
const double focal_length = 594.2944545178186;
const cv::Size show_size = {444, 250};

template <typename T>
cv::Mat clipMatValues(const cv::Mat &inputMat, T minValue, T maxValue) {
  cv::Mat response;
  inputMat.copyTo(response);
  cv::MatIterator_<T> it;
  cv::MatIterator_<T> end;
  for (it = response.begin<T>(), end = response.end<T>(); it != end; ++it) {
    if (*it < minValue) {
      *it = minValue;
    } else if (*it > maxValue) {
      *it = maxValue;
    }
  }
  return response;
}

cv::Mat ImgToGradient(const cv::Mat &img, const double min, const double max) {

  cv::Mat img_float;
  img.convertTo(img_float, CV_32F);
  const cv::Mat imgScaled = (img_float - min) / (max - min);
  cv::Mat grayGradient = clipMatValues<float>(imgScaled, 0, 1);
  grayGradient *= 255;
  (grayGradient).convertTo(grayGradient, CV_8UC1);
  // cv::cvtColor(grayGradient, grayGradient, cv::COLOR_GRAY2BGR);
  return grayGradient;
}
cv::Mat ImgToColorGradient(const cv::Mat &img, const double min,
                           const double max) {
  const cv::Mat imgScaled = (img - min) / (max - min);
  cv::Mat grayGradient = clipMatValues<float>(imgScaled, 0, 1);
  cv::Mat inverseGradient = -grayGradient + 1;

  grayGradient *= 255;
  inverseGradient *= 255;

  std::array<cv::Mat, 3> chanels;
  (grayGradient).convertTo(chanels[0], CV_8UC1);
  chanels[1] = cv::Mat::zeros(grayGradient.size(), CV_8UC1);
  (inverseGradient).convertTo(chanels[2], CV_8UC1);

  cv::Mat finalImage;
  cv::merge(chanels.data(), 3, finalImage);
  return finalImage;
}

cv::Mat computeDepthMap(const cv::Mat &dispMap, const double fLength,
                        const double baseline, const double scale) {
  cv::Mat dispMapFloat;
  dispMap.convertTo(dispMapFloat, CV_32F);
  dispMapFloat /= 16;
  dispMapFloat *= scale;
  cv::Mat depth = (fLength * baseline) / dispMapFloat;
  return depth;
}

/* Matcher type */
typedef enum { BM, SGBM } MatcherType;

/* Main data structure definition */
struct ChData {
  /* Widgets */
  GtkWidget *main_window; /* Main application window */
  GtkImage *image_left;
  GtkImage *image_right;
  GtkImage *image_disp;
  GtkImage *image_disp_filterd;
  GtkImage *image_depth;
  GtkImage *image_depth_filtered;

  GtkWidget *rb_bm, *rb_sgbm;
  GtkWidget *sc_block_size, *sc_min_disparity, *sc_num_disparities,
      *sc_disp_max_diff, *sc_speckle_range, *sc_speckle_window_size, *sc_p1,
      *sc_p2, *sc_pre_filter_cap, *sc_pre_filter_size, *sc_uniqueness_ratio,
      *sc_texture_threshold, *rb_pre_filter_normalized, *rb_pre_filter_xsobel,
      *chk_full_dp, *sc_scale, *sc_sigma, *sc_lambda, *sc_min_depth,
      *sc_max_depth;
  GtkAdjustment *adj_block_size, *adj_min_disparity, *adj_num_disparities,
      *adj_disp_max_diff, *adj_speckle_range, *adj_speckle_window_size, *adj_p1,
      *adj_p2, *adj_pre_filter_cap, *adj_pre_filter_size, *adj_uniqueness_ratio,
      *adj_texture_threshold, *adj_scale, *adj_sigma, *adj_lambda,
      *adj_min_depth, *adj_max_depth;
  GtkWidget *status_bar;
  gint status_bar_context;

  /* OpenCV */
  Ptr<StereoMatcher> stereo_matcher_left;
  Ptr<StereoMatcher> stereo_matcher_right;
  Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
  Mat cv_image_left, cv_image_right, cv_image_disparity_left,
      cv_image_disparity_right, cv_image_disparity_normalized, cv_color_image,
      cv_image_disp_filtered, cv_image_depth, cv_image_depth_filtered;
  MatcherType matcher_type;
  int block_size;
  int disp_12_max_diff;
  int min_disparity;
  int num_disparities;
  int speckle_range;
  int speckle_window_size;
  int pre_filter_cap;
  int pre_filter_size;
  int pre_filter_type;
  int texture_threshold;
  int uniqueness_ratio;
  int p1;
  int p2;
  int mode;
  double scale;
  double sigmaC;
  int lambda;
  int min_depth;
  int max_depth;

  Rect *roi1, *roi2;

  bool live_update;

  /* Defalt values */
  static const int DEFAULT_BLOCK_SIZE = 5;
  static const int DEFAULT_DISP_12_MAX_DIFF = -1;
  static const int DEFAULT_MIN_DISPARITY = 0;
  static const int DEFAULT_NUM_DISPARITIES = 64;
  static const int DEFAULT_SPECKLE_RANGE = 0;
  static const int DEFAULT_SPECKLE_WINDOW_SIZE = 0;
  static const int DEFAULT_PRE_FILTER_CAP = 1;
  static const int DEFAULT_PRE_FILTER_SIZE = 5;
  static const int DEFAULT_PRE_FILTER_TYPE =
      StereoBM::PREFILTER_NORMALIZED_RESPONSE;
  static const int DEFAULT_TEXTURE_THRESHOLD = 0;
  static const int DEFAULT_UNIQUENESS_RATIO = 0;
  static const int DEFAULT_P1 = 0;
  static const int DEFAULT_P2 = 0;
  static const int DEFAULT_MODE = StereoSGBM::MODE_SGBM;
  static constexpr double DEFAULT_SCALE = 2.0;
  static constexpr double DEFAULT_SIGMA = 0.5;
  static constexpr int DEFAULT_LAMBDA = 10000;
  static constexpr int DEFAULT_MIN_DEPTH = 240;
  static constexpr int DEFAULT_MAX_DEPTH = 410;

  ChData()
      : matcher_type(BM), block_size(DEFAULT_BLOCK_SIZE),
        disp_12_max_diff(DEFAULT_DISP_12_MAX_DIFF),
        min_disparity(DEFAULT_MIN_DISPARITY),
        num_disparities(DEFAULT_NUM_DISPARITIES),
        speckle_range(DEFAULT_SPECKLE_RANGE),
        speckle_window_size(DEFAULT_SPECKLE_WINDOW_SIZE),
        pre_filter_cap(DEFAULT_PRE_FILTER_CAP),
        pre_filter_size(DEFAULT_PRE_FILTER_SIZE),
        pre_filter_type(DEFAULT_PRE_FILTER_TYPE),
        texture_threshold(DEFAULT_TEXTURE_THRESHOLD),
        uniqueness_ratio(DEFAULT_UNIQUENESS_RATIO), p1(DEFAULT_P1),
        p2(DEFAULT_P2), mode(DEFAULT_MODE), roi1(NULL), scale(DEFAULT_SCALE),
        sigmaC(DEFAULT_SIGMA), lambda(DEFAULT_LAMBDA),
        min_depth(DEFAULT_MIN_DEPTH), max_depth(DEFAULT_MAX_DEPTH), roi2(NULL),
        live_update(true) {}
};

void update_interface(ChData *data, bool recompute = true);
void update_matcher(ChData *data) {
  if (!data->live_update) {
    return;
  }

  Ptr<StereoBM> stereo_bm;
  Ptr<StereoSGBM> stereo_sgbm;

  switch (data->matcher_type) {
  case BM:
    stereo_bm = data->stereo_matcher_left.dynamicCast<StereoBM>();

    // If we have the wrong type of matcher, let's create a new one:
    if (!stereo_bm) {
      data->stereo_matcher_left = stereo_bm = StereoBM::create(16, 1);

      gtk_widget_set_sensitive(data->sc_block_size, true);
      gtk_widget_set_sensitive(data->sc_min_disparity, true);
      gtk_widget_set_sensitive(data->sc_num_disparities, true);
      gtk_widget_set_sensitive(data->sc_disp_max_diff, true);
      gtk_widget_set_sensitive(data->sc_speckle_range, true);
      gtk_widget_set_sensitive(data->sc_speckle_window_size, true);
      gtk_widget_set_sensitive(data->sc_p1, false);
      gtk_widget_set_sensitive(data->sc_p2, false);
      gtk_widget_set_sensitive(data->sc_pre_filter_cap, true);
      gtk_widget_set_sensitive(data->sc_pre_filter_size, true);
      gtk_widget_set_sensitive(data->sc_uniqueness_ratio, true);
      gtk_widget_set_sensitive(data->sc_texture_threshold, true);
      gtk_widget_set_sensitive(data->sc_scale, true);
      gtk_widget_set_sensitive(data->rb_pre_filter_normalized, true);
      gtk_widget_set_sensitive(data->rb_pre_filter_xsobel, true);
      gtk_widget_set_sensitive(data->chk_full_dp, false);
    }

    stereo_bm->setBlockSize(data->block_size);
    stereo_bm->setDisp12MaxDiff(data->disp_12_max_diff);
    stereo_bm->setMinDisparity(data->min_disparity);
    stereo_bm->setNumDisparities(data->num_disparities);
    stereo_bm->setSpeckleRange(data->speckle_range);
    stereo_bm->setSpeckleWindowSize(data->speckle_window_size);
    stereo_bm->setPreFilterCap(data->pre_filter_cap);
    stereo_bm->setPreFilterSize(data->pre_filter_size);
    stereo_bm->setPreFilterType(data->pre_filter_type);
    stereo_bm->setTextureThreshold(data->texture_threshold);
    stereo_bm->setUniquenessRatio(data->uniqueness_ratio);
    data->stereo_matcher_right =
        cv::ximgproc::createRightMatcher(data->stereo_matcher_left);
    data->wls_filter =
        cv::ximgproc::createDisparityWLSFilter(data->stereo_matcher_left);
    data->wls_filter->setLambda(data->lambda);
    data->wls_filter->setSigmaColor(data->sigmaC);

    if (data->roi1 != NULL && data->roi2 != NULL) {
      stereo_bm->setROI1(*data->roi1);
      stereo_bm->setROI2(*data->roi2);
    }
    break;

  case SGBM:
    stereo_sgbm = data->stereo_matcher_left.dynamicCast<StereoSGBM>();

    // If we have the wrong type of matcher, let's create a new one:
    if (!stereo_sgbm) {
      data->stereo_matcher_left = stereo_sgbm = StereoSGBM::create(
          ChData::DEFAULT_MIN_DISPARITY, ChData::DEFAULT_NUM_DISPARITIES,
          ChData::DEFAULT_BLOCK_SIZE, ChData::DEFAULT_P1, ChData::DEFAULT_P2,
          ChData::DEFAULT_DISP_12_MAX_DIFF, ChData::DEFAULT_PRE_FILTER_CAP,
          ChData::DEFAULT_UNIQUENESS_RATIO, ChData::DEFAULT_SPECKLE_WINDOW_SIZE,
          ChData::DEFAULT_SPECKLE_RANGE, ChData::DEFAULT_MODE);

      gtk_widget_set_sensitive(data->sc_block_size, true);
      gtk_widget_set_sensitive(data->sc_min_disparity, true);
      gtk_widget_set_sensitive(data->sc_num_disparities, true);
      gtk_widget_set_sensitive(data->sc_disp_max_diff, true);
      gtk_widget_set_sensitive(data->sc_speckle_range, true);
      gtk_widget_set_sensitive(data->sc_speckle_window_size, true);
      gtk_widget_set_sensitive(data->sc_p1, true);
      gtk_widget_set_sensitive(data->sc_p2, true);
      gtk_widget_set_sensitive(data->sc_pre_filter_cap, true);
      gtk_widget_set_sensitive(data->sc_pre_filter_size, false);
      gtk_widget_set_sensitive(data->sc_uniqueness_ratio, true);
      gtk_widget_set_sensitive(data->sc_texture_threshold, false);
      gtk_widget_set_sensitive(data->sc_scale, true);
      gtk_widget_set_sensitive(data->rb_pre_filter_normalized, false);
      gtk_widget_set_sensitive(data->rb_pre_filter_xsobel, false);
      gtk_widget_set_sensitive(data->chk_full_dp, true);
    }

    stereo_sgbm->setBlockSize(data->block_size);
    std::cout << "\t\t\tchaning disp12 " << data->disp_12_max_diff << std::endl;
    stereo_sgbm->setDisp12MaxDiff(data->disp_12_max_diff);
    std::cout << "\t\t\tread disp12 " << stereo_sgbm->getDisp12MaxDiff()
              << std::endl;
    stereo_sgbm->setMinDisparity(data->min_disparity);
    stereo_sgbm->setMode(data->mode);
    stereo_sgbm->setNumDisparities(data->num_disparities);
    stereo_sgbm->setP1(data->p1);
    stereo_sgbm->setP2(data->p2);
    stereo_sgbm->setPreFilterCap(data->pre_filter_cap);
    stereo_sgbm->setSpeckleRange(data->speckle_range);
    stereo_sgbm->setSpeckleWindowSize(data->speckle_window_size);
    stereo_sgbm->setUniquenessRatio(data->uniqueness_ratio);
    stereo_sgbm->setMode(cv::StereoSGBM::MODE_HH4);
    // stereo_sgbm = data->stereo_matcher_left.dynamicCast<StereoSGBM>();
    if (data->min_depth < 0) {
      data->min_depth = ChData::DEFAULT_MIN_DEPTH;
    }
    const double max_disp =
        focal_length * baseline / data->min_depth / data->scale;
    const double min_disp =
        focal_length * baseline / data->max_depth / data->scale;
    const int num_disp = int(std::ceil((max_disp - min_disp) / 16.0)) * 16;

    std::cout << "num_disp " << num_disp << " min disp " << min_disp
              << " max disp " << max_disp << std::endl;
    std::cout << "min_depth" << data->min_depth << " max disp "
              << data->max_depth << std::endl;
    stereo_sgbm->setMinDisparity(int(min_disp));
    stereo_sgbm->setNumDisparities(num_disp);

    data->stereo_matcher_right =
        cv::ximgproc::createRightMatcher(data->stereo_matcher_left);
    data->wls_filter =
        cv::ximgproc::createDisparityWLSFilter(data->stereo_matcher_left);
    data->wls_filter->setLambda(data->lambda);
    data->wls_filter->setSigmaColor(data->sigmaC);
    update_interface(data, false);

    break;
  }

  clock_t t;
  t = clock();
  std::cout << "computing disparity map\n";
  cv::Mat left_image_scaled;
  cv::Mat right_image_scaled;

  cv::resize(data->cv_image_left, left_image_scaled, cv::Size{},
             1 / data->scale, 1 / data->scale, cv::INTER_AREA);
  cv::resize(data->cv_image_right, right_image_scaled, cv::Size{},
             1 / data->scale, 1 / data->scale, cv::INTER_AREA);
  data->stereo_matcher_left->compute(left_image_scaled, right_image_scaled,
                                     data->cv_image_disparity_left);
  data->stereo_matcher_right->compute(right_image_scaled, left_image_scaled,
                                      data->cv_image_disparity_right);

  data->cv_image_disparity_left.convertTo(data->cv_image_disparity_left,
                                          CV_16S);
  data->cv_image_disparity_right.convertTo(data->cv_image_disparity_right,
                                           CV_16S);

  data->wls_filter->filter(data->cv_image_disparity_left, left_image_scaled,
                           data->cv_image_disp_filtered,
                           data->cv_image_disparity_right, cv::Rect{},
                           right_image_scaled);

  cv::Mat depthMapFiltered = computeDepthMap(
      data->cv_image_disp_filtered, focal_length, baseline, data->scale);
  cv::Mat depthMap = computeDepthMap(data->cv_image_disparity_left,
                                     focal_length, baseline, data->scale);
  // color gradient for disp map
  const double max_disp =
      focal_length * baseline / data->min_depth * 16 / data->scale;
  const double min_disp =
      focal_length * baseline / data->max_depth * 16 / data->scale;

  cv::Mat dispCheckMask;
  cv::Mat leftDispMapFloat;
  double min, max;
  cv::minMaxLoc(data->cv_image_disparity_left, &min, &max);
  data->cv_image_disparity_left.convertTo(leftDispMapFloat, CV_32F);
  cv::threshold(leftDispMapFloat, dispCheckMask, min_disp, 255,
                THRESH_BINARY_INV);
  dispCheckMask.convertTo(dispCheckMask, CV_8UC1);
  cv::imshow("mask", dispCheckMask);
  std::cout << "\t\tmin " << min << "  " << int(min_disp) << " "
            << data->stereo_matcher_left->getDisp12MaxDiff() << std::endl;
  cv::waitKey(1);
  data->cv_image_disparity_left =
      ImgToGradient(data->cv_image_disparity_left, min_disp, max_disp);
  data->cv_image_disp_filtered =
      ImgToGradient(data->cv_image_disp_filtered, min_disp, max_disp);
  // color gradient for depth map
  data->cv_image_depth_filtered =
      ImgToGradient(depthMapFiltered, data->min_depth, data->max_depth);
  data->cv_image_depth =
      ImgToGradient(depthMap, data->min_depth, data->max_depth);

  std::cout << "computed disparity map\n";
  t = clock() - t;

  gchar *status_message =
      g_strdup_printf("Disparity computation took %lf milliseconds",
                      ((double)t * 1000) / CLOCKS_PER_SEC);
  gtk_statusbar_pop(GTK_STATUSBAR(data->status_bar), data->status_bar_context);
  gtk_statusbar_push(GTK_STATUSBAR(data->status_bar), data->status_bar_context,
                     status_message);
  g_free(status_message);

  // normalize(data->cv_image_disparity_left,
  // data->cv_image_disparity_normalized,
  //           0, 255, cv::NORM_MINMAX, CV_8UC1);
  // cvtColor(data->cv_image_disparity_normalized, data->cv_color_image,
  //          cv::COLOR_GRAY2BGR);
  // cv::resize(data->cv_color_image, data->cv_color_image, show_size);

  // resize all images that will be shown (disparites and depth maps)
  std::cout << "resizing images\n";
  cv::resize(data->cv_image_disparity_left, data->cv_image_disparity_left,
             show_size, 0, 0, cv::INTER_NEAREST_EXACT);
  cv::resize(data->cv_image_disp_filtered, data->cv_image_disp_filtered,
             show_size, 0, 0, cv::INTER_NEAREST_EXACT);
  cv::resize(data->cv_image_depth, data->cv_image_depth, show_size, 0, 0,
             cv::INTER_NEAREST_EXACT);
  cv::resize(data->cv_image_depth_filtered, data->cv_image_depth_filtered,
             show_size, 0, 0, cv::INTER_NEAREST_EXACT);
  // make all images BRG
  std::cout << "converting color disp\n";
  cv::cvtColor(data->cv_image_disparity_left, data->cv_image_disparity_left,
               cv::COLOR_GRAY2BGR);
  std::cout << "converting color disp filtered\n";
  cv::cvtColor(data->cv_image_disp_filtered, data->cv_image_disp_filtered,
               cv::COLOR_GRAY2BGR);
  std::cout << "converting color depth \n";
  cv::cvtColor(data->cv_image_depth, data->cv_image_depth, cv::COLOR_GRAY2BGR);
  // cv::cvtColor(data->cv_image_depth, data->cv_image_depth,
  // cv::COLOR_GRAY2BGR);
  cv::cvtColor(data->cv_image_depth_filtered, data->cv_image_depth_filtered,
               cv::COLOR_GRAY2BGR);
  std::cout << "done converting color\n";

  const auto img_to_pix_buff = [](const cv::Mat &img) -> GdkPixbuf * {
    return gdk_pixbuf_new_from_data((guchar *)img.data, GDK_COLORSPACE_RGB,
                                    false, 8, img.cols, img.rows, img.step,
                                    NULL, NULL);
  };
  auto *pixbuff_disp = img_to_pix_buff(data->cv_image_disparity_left);
  auto *pixbuff_disp_filtered = img_to_pix_buff(data->cv_image_disp_filtered);
  auto *pixbuff_depth = img_to_pix_buff(data->cv_image_depth);
  auto *pixbuff_depth_filtered = img_to_pix_buff(data->cv_image_depth_filtered);
  // GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
  //     (guchar *)data->cv_color_image.data, GDK_COLORSPACE_RGB, false, 8,
  //     data->cv_color_image.cols, data->cv_color_image.rows,
  //     data->cv_color_image.step, NULL, NULL);
  gtk_image_set_from_pixbuf(data->image_disp, pixbuff_disp);
  gtk_image_set_from_pixbuf(data->image_disp_filterd, pixbuff_disp_filtered);
  gtk_image_set_from_pixbuf(data->image_depth, pixbuff_depth);
  gtk_image_set_from_pixbuf(data->image_depth_filtered, pixbuff_depth_filtered);
}

void update_interface(ChData *data, bool recompute) {
  // Avoids rebuilding the matcher on every change:
  data->live_update = false;

  if (data->matcher_type == BM) {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_bm), true);
  } else {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_sgbm), true);
  }

  gtk_adjustment_set_value(data->adj_block_size, data->block_size);
  gtk_adjustment_set_value(data->adj_min_disparity, data->min_disparity);
  gtk_adjustment_set_value(data->adj_num_disparities, data->num_disparities);
  gtk_adjustment_set_value(data->adj_disp_max_diff, data->disp_12_max_diff);
  gtk_adjustment_set_value(data->adj_speckle_range, data->speckle_range);
  gtk_adjustment_set_value(data->adj_speckle_window_size,
                           data->speckle_window_size);
  gtk_adjustment_set_value(data->adj_p1, data->p1);
  gtk_adjustment_set_value(data->adj_p2, data->p2);
  gtk_adjustment_set_value(data->adj_pre_filter_cap, data->pre_filter_cap);
  gtk_adjustment_set_value(data->adj_pre_filter_size, data->pre_filter_size);
  gtk_adjustment_set_value(data->adj_uniqueness_ratio, data->uniqueness_ratio);
  gtk_adjustment_set_value(data->adj_texture_threshold,
                           data->texture_threshold);
  gtk_adjustment_set_value(data->adj_scale, data->scale);
  gtk_adjustment_set_value(data->adj_sigma, data->sigmaC);
  gtk_adjustment_set_value(data->adj_lambda, data->lambda);
  gtk_adjustment_set_value(data->adj_min_depth, data->min_depth);
  gtk_adjustment_set_value(data->adj_max_depth, data->max_depth);

  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->chk_full_dp),
                               data->mode == StereoSGBM::MODE_HH);

  if (data->pre_filter_type == StereoBM::PREFILTER_NORMALIZED_RESPONSE) {
    gtk_toggle_button_set_active(
        GTK_TOGGLE_BUTTON(data->rb_pre_filter_normalized), true);
  } else {
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(data->rb_pre_filter_xsobel),
                                 true);
  }

  data->live_update = true;
  if (recompute) {
    update_matcher(data);
  }
}

extern "C" {
G_MODULE_EXPORT void on_adj_block_size_value_changed(GtkAdjustment *adjustment,
                                                     ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  // the value must be odd, if it is not then set it to the next odd value
  if (value % 2 == 0) {
    value += 1;
    gtk_adjustment_set_value(adjustment, (gdouble)value);
    return;
  }

  // the value must be smaller than the image size
  if (value >= data->cv_image_left.cols || value >= data->cv_image_left.rows) {
    fprintf(stderr, "WARNING: Block size is larger than image size\n");
    return;
  }

  // set the parameter,
  data->block_size = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_min_disparity_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->min_disparity = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_num_disparities_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  // te value must be divisible by 16, if it is not set it to the nearest
  // multiple of 16
  if (value % 16 != 0) {
    value += (16 - value % 16);
    gtk_adjustment_set_value(adjustment, (gdouble)value);
    return;
  }

  data->num_disparities = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_disp_max_diff_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->disp_12_max_diff = value;
  std::cout << "chaning disp12 " << value << std::endl;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_speckle_range_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->speckle_range = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_speckle_window_size_value_changed(GtkAdjustment *adjustment,
                                         ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->speckle_window_size = value;
  update_matcher(data);
}

G_MODULE_EXPORT void on_adj_p1_value_changed(GtkAdjustment *adjustment,
                                             ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->p1 = value;
  update_matcher(data);
}

G_MODULE_EXPORT void on_adj_p2_value_changed(GtkAdjustment *adjustment,
                                             ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->p2 = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_pre_filter_cap_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  // set the parameter
  data->pre_filter_cap = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_pre_filter_size_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  // the value must be odd, if it is not then set it to the next odd value
  if (value % 2 == 0) {
    value += 1;
    gtk_adjustment_set_value(adjustment, (gdouble)value);
    return;
  }

  // set the parameter,
  data->pre_filter_size = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_uniqueness_ratio_value_changed(GtkAdjustment *adjustment, ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->uniqueness_ratio = value;
  update_matcher(data);
}

G_MODULE_EXPORT void
on_adj_texture_threshold_value_changed(GtkAdjustment *adjustment,
                                       ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->texture_threshold = value;
  update_matcher(data);
}
G_MODULE_EXPORT void on_adj_scale_value_changed(GtkAdjustment *adjustment,
                                                ChData *data) {
  gdouble value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gdouble)gtk_adjustment_get_value(adjustment);

  data->scale = value;
  update_matcher(data);
}
G_MODULE_EXPORT void on_adj_sigma_value_changed(GtkAdjustment *adjustment,
                                                ChData *data) {
  gdouble value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gdouble)gtk_adjustment_get_value(adjustment);

  data->sigmaC = value;
  update_matcher(data);
}
G_MODULE_EXPORT void on_adj_lambda_value_changed(GtkAdjustment *adjustment,
                                                 ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->lambda = value;
  std::cout << "\t changed lambda " << data->lambda << std::endl;
  update_matcher(data);
}
G_MODULE_EXPORT void on_adj_min_depth_value_changed(GtkAdjustment *adjustment,
                                                    ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->min_depth = value;
  update_matcher(data);
}
G_MODULE_EXPORT void on_adj_max_depth_value_changed(GtkAdjustment *adjustment,
                                                    ChData *data) {
  gint value;

  if (data == NULL) {
    fprintf(stderr, "WARNING: data is null\n");
    return;
  }

  value = (gint)gtk_adjustment_get_value(adjustment);

  data->max_depth = value;
  update_matcher(data);
}

G_MODULE_EXPORT void on_algo_ssgbm_clicked(GtkButton *b, ChData *data) {
  if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
    data->matcher_type = SGBM;
    update_matcher(data);
  }
}

G_MODULE_EXPORT void on_algo_sbm_clicked(GtkButton *b, ChData *data) {
  if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
    data->matcher_type = BM;
    update_matcher(data);
  }
}

G_MODULE_EXPORT void on_rb_pre_filter_normalized_clicked(GtkButton *b,
                                                         ChData *data) {
  if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
    data->pre_filter_type = StereoBM::PREFILTER_NORMALIZED_RESPONSE;
    update_matcher(data);
  }
}

G_MODULE_EXPORT void on_rb_pre_filter_xsobel_clicked(GtkButton *b,
                                                     ChData *data) {
  if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
    data->pre_filter_type = StereoBM::PREFILTER_XSOBEL;
    update_matcher(data);
  }
}

G_MODULE_EXPORT void on_chk_full_dp_clicked(GtkButton *b, ChData *data) {
  if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(b))) {
    data->mode = StereoSGBM::MODE_HH;
  } else {
    data->mode = StereoSGBM::MODE_SGBM;
  }
  update_matcher(data);
}

G_MODULE_EXPORT void on_btn_save_clicked(GtkButton *b, ChData *data) {
  GtkWidget *dialog;
  GtkFileChooser *chooser;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_SAVE;
  gint res;

  dialog = gtk_file_chooser_dialog_new(
      "Save File", GTK_WINDOW(data->main_window), action, "Cancel",
      GTK_RESPONSE_CANCEL, "Save", GTK_RESPONSE_ACCEPT, NULL);
  chooser = GTK_FILE_CHOOSER(dialog);
  gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE);
  gtk_file_chooser_set_current_name(chooser, "params.yml");

  GtkFileFilter *filter_yml = gtk_file_filter_new();
  gtk_file_filter_set_name(filter_yml, "YAML file (*.yml)");
  gtk_file_filter_add_pattern(filter_yml, "*.yml");

  GtkFileFilter *filter_xml = gtk_file_filter_new();
  gtk_file_filter_set_name(filter_xml, "XML file(*.xml)");
  gtk_file_filter_add_pattern(filter_xml, "*.xml");

  gtk_file_chooser_add_filter(chooser, filter_yml);
  gtk_file_chooser_add_filter(chooser, filter_xml);

  res = gtk_dialog_run(GTK_DIALOG(dialog));
  char *filename;
  filename = gtk_file_chooser_get_filename(chooser);
  gtk_widget_destroy(GTK_WIDGET(dialog));

  if (res == GTK_RESPONSE_ACCEPT) {
    int len = strlen(filename);

    if (!strcmp(filename + len - 4, ".yml") ||
        !strcmp(filename + len - 4, ".xml")) {
      FileStorage fs(filename, FileStorage::WRITE);

      switch (data->matcher_type) {
      case BM:
        fs << "name"
           << "StereoMatcher.BM"
           << "blockSize" << data->block_size << "minDisparity"
           << data->min_disparity << "numDisparities" << data->num_disparities
           << "disp12MaxDiff" << data->disp_12_max_diff << "speckleRange"
           << data->speckle_range << "speckleWindowSize"
           << data->speckle_window_size << "preFilterCap"
           << data->pre_filter_cap << "preFilterSize" << data->pre_filter_size
           << "uniquenessRatio" << data->uniqueness_ratio << "textureThreshold"
           << data->texture_threshold << "preFilterType"
           << data->pre_filter_type;
        break;

      case SGBM:
        fs << "name"
           << "StereoMatcher.SGBM"
           << "blockSize" << data->block_size << "minDisparity"
           << data->min_disparity << "numDisparities" << data->num_disparities
           << "disp12MaxDiff" << data->disp_12_max_diff << "speckleRange"
           << data->speckle_range << "speckleWindowSize"
           << data->speckle_window_size << "P1" << data->p1 << "P2" << data->p2
           << "preFilterCap" << data->pre_filter_cap << "uniquenessRatio"
           << data->uniqueness_ratio << "mode" << data->mode;
        break;
      }
      fs.release();

      GtkWidget *message = gtk_message_dialog_new(
          GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
          GTK_MESSAGE_INFO, GTK_BUTTONS_CLOSE, "Parameters saved successfully");
      gtk_dialog_run(GTK_DIALOG(message));
      gtk_widget_destroy(GTK_WIDGET(message));
    } else {
      GtkWidget *message = gtk_message_dialog_new(
          GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
          GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE,
          "Currently the only supported formats are XML and YAML.");
      gtk_dialog_run(GTK_DIALOG(message));
      gtk_widget_destroy(GTK_WIDGET(message));
    }

    g_free(filename);
  }
}

G_MODULE_EXPORT void on_btn_load_clicked(GtkButton *b, ChData *data) {
  GtkWidget *dialog;
  GtkFileChooser *chooser;
  GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
  gint res;

  dialog = gtk_file_chooser_dialog_new(
      "Open File", GTK_WINDOW(data->main_window), action, "Cancel",
      GTK_RESPONSE_CANCEL, "Open", GTK_RESPONSE_ACCEPT, NULL);
  chooser = GTK_FILE_CHOOSER(dialog);

  GtkFileFilter *filter = gtk_file_filter_new();
  gtk_file_filter_set_name(filter, "YAML or XML file (*.yml, *.xml)");
  gtk_file_filter_add_pattern(filter, "*.yml");
  gtk_file_filter_add_pattern(filter, "*.xml");

  gtk_file_chooser_add_filter(chooser, filter);

  res = gtk_dialog_run(GTK_DIALOG(dialog));
  char *filename;
  filename = gtk_file_chooser_get_filename(chooser);
  gtk_widget_destroy(GTK_WIDGET(dialog));

  if (res == GTK_RESPONSE_ACCEPT) {
    int len = strlen(filename);

    if (!strcmp(filename + len - 4, ".yml") ||
        !strcmp(filename + len - 4, ".xml")) {
      FileStorage fs(filename, FileStorage::READ);

      if (!fs.isOpened()) {
        GtkWidget *message = gtk_message_dialog_new(
            GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
            GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE,
            "Could not open the selected file.");
        gtk_dialog_run(GTK_DIALOG(message));
        gtk_widget_destroy(GTK_WIDGET(message));
      } else {
        string name;
        fs["name"] >> name;

        if (name == "StereoMatcher.BM") {
          data->matcher_type = BM;
          fs["blockSize"] >> data->block_size;
          fs["minDisparity"] >> data->min_disparity;
          fs["numDisparities"] >> data->num_disparities;
          fs["disp12MaxDiff"] >> data->disp_12_max_diff;
          fs["speckleRange"] >> data->speckle_range;
          fs["speckleWindowSize"] >> data->speckle_window_size;
          fs["preFilterCap"] >> data->pre_filter_cap;
          fs["preFilterSize"] >> data->pre_filter_size;
          fs["uniquenessRatio"] >> data->uniqueness_ratio;
          fs["textureThreshold"] >> data->texture_threshold;
          fs["preFilterType"] >> data->pre_filter_type;
          update_interface(data);

          GtkWidget *message = gtk_message_dialog_new(
              GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
              GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE,
              "Parameters loaded successfully.");
          gtk_dialog_run(GTK_DIALOG(message));
          gtk_widget_destroy(GTK_WIDGET(message));
        } else if (name == "StereoMatcher.SGBM") {
          data->matcher_type = SGBM;
          fs["blockSize"] >> data->block_size;
          fs["minDisparity"] >> data->min_disparity;
          fs["numDisparities"] >> data->num_disparities;
          fs["disp12MaxDiff"] >> data->disp_12_max_diff;
          fs["speckleRange"] >> data->speckle_range;
          fs["speckleWindowSize"] >> data->speckle_window_size;
          fs["P1"] >> data->p1;
          fs["P2"] >> data->p2;
          fs["preFilterCap"] >> data->pre_filter_cap;
          fs["uniquenessRatio"] >> data->uniqueness_ratio;
          fs["mode"] >> data->mode;
          update_interface(data);

          GtkWidget *message = gtk_message_dialog_new(
              GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
              GTK_MESSAGE_INFO, GTK_BUTTONS_CLOSE,
              "Parameters loaded successfully.");
          gtk_dialog_run(GTK_DIALOG(message));
          gtk_widget_destroy(GTK_WIDGET(message));
        } else {
          GtkWidget *message = gtk_message_dialog_new(
              GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
              GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE, "This file is not valid.");
          gtk_dialog_run(GTK_DIALOG(message));
          gtk_widget_destroy(GTK_WIDGET(message));
        }

        fs.release();
      }
    } else {
      GtkWidget *message = gtk_message_dialog_new(
          GTK_WINDOW(data->main_window), GTK_DIALOG_DESTROY_WITH_PARENT,
          GTK_MESSAGE_ERROR, GTK_BUTTONS_CLOSE,
          "Currently the only supported formats are XML and YAML.");
      gtk_dialog_run(GTK_DIALOG(message));
      gtk_widget_destroy(GTK_WIDGET(message));
    }

    g_free(filename);
  }
}

G_MODULE_EXPORT void on_btn_defaults_clicked(GtkButton *b, ChData *data) {
  data->matcher_type = SGBM;
  data->block_size = ChData::DEFAULT_BLOCK_SIZE;
  data->disp_12_max_diff = ChData::DEFAULT_DISP_12_MAX_DIFF;
  data->min_disparity = ChData::DEFAULT_MIN_DISPARITY;
  data->num_disparities = ChData::DEFAULT_NUM_DISPARITIES;
  data->speckle_range = ChData::DEFAULT_SPECKLE_RANGE;
  data->speckle_window_size = ChData::DEFAULT_SPECKLE_WINDOW_SIZE;
  data->pre_filter_cap = ChData::DEFAULT_PRE_FILTER_CAP;
  data->pre_filter_size = ChData::DEFAULT_PRE_FILTER_SIZE;
  data->pre_filter_type = ChData::DEFAULT_PRE_FILTER_TYPE;
  data->texture_threshold = ChData::DEFAULT_TEXTURE_THRESHOLD;
  data->uniqueness_ratio = ChData::DEFAULT_UNIQUENESS_RATIO;
  data->p1 = ChData::DEFAULT_P1;
  data->p2 = ChData::DEFAULT_P2;
  data->mode = ChData::DEFAULT_MODE;
  data->sigmaC = ChData::DEFAULT_SIGMA;
  data->lambda = ChData::DEFAULT_LAMBDA;
  data->min_depth = ChData::DEFAULT_MIN_DEPTH;
  data->max_depth = ChData::DEFAULT_MAX_DEPTH;
  data->scale = ChData::DEFAULT_SCALE;
  update_interface(data);
}
}

int main(int argc, char *argv[]) {
  char default_left_filename[] = "tsukuba/scene1.row3.col3.ppm";
  char default_right_filename[] = "tsukuba/scene1.row3.col5.ppm";
  char *left_filename = default_left_filename;
  char *right_filename = default_right_filename;
  char *extrinsics_filename = NULL;
  char *intrinsics_filename = NULL;

  GtkBuilder *builder;
  GError *error = NULL;
  ChData *data;

  /* Parse arguments to find left and right filenames */
  // TODO: we should use some library to parse the command line arguments if we
  // are going to use lots of them.
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-left") == 0) {
      i++;
      left_filename = argv[i];
    } else if (strcmp(argv[i], "-right") == 0) {
      i++;
      right_filename = argv[i];
    } else if (strcmp(argv[i], "-extrinsics") == 0) {
      i++;
      extrinsics_filename = argv[i];
    } else if (strcmp(argv[i], "-intrinsics") == 0) {
      i++;
      intrinsics_filename = argv[i];
    }
  }

  Mat left_image = imread(left_filename, 1);
  // left_image *= 6.;

  if (left_image.empty()) {
    printf("Could not read left image %s.\n", left_filename);
    exit(1);
  }

  Mat right_image = imread(right_filename, 1);
  // right_image *= 6;

  if (right_image.empty()) {
    printf("Could not read right image %s.\n", right_filename);
    exit(1);
  }

  if (left_image.size() != right_image.size()) {
    printf("Left and right images have different sizes.\n");
    exit(1);
  }
  std::cout << "images where loaded correctly\n";

  Mat gray_left, gray_right;
  cvtColor(left_image, gray_left, cv::COLOR_BGR2GRAY);
  cvtColor(right_image, gray_right, cv::COLOR_BGR2GRAY);
  std::cout << "images where converted correctly\n";

  /* Create data */
  data = new ChData();

  if (intrinsics_filename != NULL && extrinsics_filename != NULL) {
    FileStorage intrinsicsFs(intrinsics_filename, FileStorage::READ);

    if (!intrinsicsFs.isOpened()) {
      printf("Could not open intrinsic parameters file %s.\n",
             intrinsics_filename);
      exit(1);
    }

    Mat m1, d1, m2, d2;
    intrinsicsFs["M1"] >> m1;
    intrinsicsFs["D1"] >> d1;
    intrinsicsFs["M2"] >> m2;
    intrinsicsFs["D2"] >> d2;

    FileStorage extrinsicsFs(extrinsics_filename, FileStorage::READ);

    if (!extrinsicsFs.isOpened()) {
      printf("Could not open extrinsic parameters file %s.\n",
             extrinsics_filename);
      exit(1);
    }

    printf(
        "Using provided calibration files to undistort and rectify images.\n");

    Mat r, t;
    extrinsicsFs["R"] >> r;
    extrinsicsFs["T"] >> t;

    Mat r1, p1, r2, p2, q;
    data->roi1 = new Rect();
    data->roi2 = new Rect();
    stereoRectify(m1, d1, m2, d2, left_image.size(), r, t, r1, r2, p1, p2, q,
                  CALIB_ZERO_DISPARITY, -1, left_image.size(), data->roi1,
                  data->roi2);

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(m1, d1, r1, p1, left_image.size(), CV_16SC2, map11,
                            map12);
    initUndistortRectifyMap(m2, d2, r2, p2, right_image.size(), CV_16SC2, map21,
                            map22);

    Mat remapped_left, remapped_right;
    remap(gray_left, remapped_left, map11, map12, INTER_LINEAR);
    remap(gray_right, remapped_right, map21, map22, INTER_LINEAR);

    data->cv_image_left = remapped_left;
    data->cv_image_right = remapped_right;

    Mat color_remapped_left, color_remapped_right;
    remap(left_image, color_remapped_left, map11, map12, INTER_LINEAR);
    remap(right_image, color_remapped_right, map21, map22, INTER_LINEAR);
    left_image = color_remapped_left;
    right_image = color_remapped_right;
  } else {
    data->cv_image_left = gray_left;
    data->cv_image_right = gray_right;
  }

  /* Init GTK+ */
  gtk_init(&argc, &argv);

  /* Create new GtkBuilder object */
  builder = gtk_builder_new();

  std::cout << "loading glade file\n";
  if (!gtk_builder_add_from_file(builder, "StereoTuner.glade", &error)) {
    std::cout << "Failed to load glade file\n";
    g_warning("%s", error->message);
    g_free(error);
    return (1);
  }
  std::cout << "loaded glade file\n";

  /* Get main window pointer from UI */
  data->main_window = GTK_WIDGET(gtk_builder_get_object(builder, "window1"));
  data->image_left = GTK_IMAGE(gtk_builder_get_object(builder, "image_left"));
  data->image_right = GTK_IMAGE(gtk_builder_get_object(builder, "image_right"));
  data->image_disp =
      GTK_IMAGE(gtk_builder_get_object(builder, "image_disparity"));
  data->image_disp_filterd =
      GTK_IMAGE(gtk_builder_get_object(builder, "image_disparity_filtered"));
  data->image_depth = GTK_IMAGE(gtk_builder_get_object(builder, "image_depth"));
  data->image_depth_filtered =
      GTK_IMAGE(gtk_builder_get_object(builder, "image_depth_filtered"));

  data->sc_block_size =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_block_size"));
  data->sc_min_disparity =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_min_disparity"));
  data->sc_num_disparities =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_num_disparities"));
  data->sc_disp_max_diff =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_disp_max_diff"));
  data->sc_speckle_range =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_range"));
  data->sc_speckle_window_size =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_speckle_window_size"));
  data->sc_p1 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p1"));
  data->sc_p2 = GTK_WIDGET(gtk_builder_get_object(builder, "sc_p2"));
  data->sc_pre_filter_cap =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_cap"));
  data->sc_pre_filter_size =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_pre_filter_size"));
  data->sc_uniqueness_ratio =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_uniqueness_ratio"));
  data->sc_texture_threshold =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_texture_threshold"));
  data->sc_scale = GTK_WIDGET(gtk_builder_get_object(builder, "sc_scale"));
  data->sc_sigma = GTK_WIDGET(gtk_builder_get_object(builder, "sc_sigma"));
  data->sc_lambda = GTK_WIDGET(gtk_builder_get_object(builder, "sc_lambda"));
  data->sc_min_depth =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_min_depth"));
  data->sc_max_depth =
      GTK_WIDGET(gtk_builder_get_object(builder, "sc_max_depth"));
  data->rb_pre_filter_normalized =
      GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_normalized"));
  data->rb_pre_filter_xsobel =
      GTK_WIDGET(gtk_builder_get_object(builder, "rb_pre_filter_xsobel"));
  data->chk_full_dp =
      GTK_WIDGET(gtk_builder_get_object(builder, "chk_full_dp"));
  data->status_bar = GTK_WIDGET(gtk_builder_get_object(builder, "status_bar"));
  data->rb_bm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_sbm"));
  data->rb_sgbm = GTK_WIDGET(gtk_builder_get_object(builder, "algo_ssgbm"));
  data->adj_block_size =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_block_size"));
  data->adj_min_disparity =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_min_disparity"));
  data->adj_num_disparities =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_num_disparities"));
  data->adj_disp_max_diff =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_disp_max_diff"));
  data->adj_speckle_range =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_speckle_range"));
  data->adj_speckle_window_size = GTK_ADJUSTMENT(
      gtk_builder_get_object(builder, "adj_speckle_window_size"));
  data->adj_p1 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p1"));
  data->adj_p2 = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_p2"));
  data->adj_pre_filter_cap =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_cap"));
  data->adj_pre_filter_size =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_pre_filter_size"));
  data->adj_uniqueness_ratio =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_uniqueness_ratio"));
  data->adj_texture_threshold =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_texture_threshold"));
  data->status_bar_context = gtk_statusbar_get_context_id(
      GTK_STATUSBAR(data->status_bar), "Statusbar context");
  data->adj_scale =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_scale"));
  data->adj_sigma =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_sigma"));
  data->adj_lambda =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_lambda"));
  data->adj_min_depth =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_min_depth"));
  data->adj_max_depth =
      GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adj_max_depth"));

  // Put images in place:
  // gtk_image_set_from_file(data->image_left, left_filename);
  // gtk_image_set_from_file(data->image_right, right_filename);

  std::cout << "setting RGB image for UI\n";
  Mat leftRGB, rightRGB;
  cvtColor(left_image, leftRGB, cv::COLOR_BGR2RGB);
  cv::resize(leftRGB, leftRGB, show_size);
  GdkPixbuf *pixbuf = gdk_pixbuf_new_from_data(
      (guchar *)leftRGB.data, GDK_COLORSPACE_RGB, false, 8, leftRGB.cols,
      leftRGB.rows, leftRGB.step, NULL, NULL);
  gtk_image_set_from_pixbuf(data->image_left, pixbuf);

  cvtColor(right_image, rightRGB, cv::COLOR_BGR2RGB);
  cv::resize(rightRGB, rightRGB, show_size);
  pixbuf = gdk_pixbuf_new_from_data((guchar *)rightRGB.data, GDK_COLORSPACE_RGB,
                                    false, 8, rightRGB.cols, rightRGB.rows,
                                    rightRGB.step, NULL, NULL);
  gtk_image_set_from_pixbuf(data->image_right, pixbuf);

  std::cout << "set RGB image for UI\n";
  update_matcher(data);

  /* Connect signals */
  gtk_builder_connect_signals(builder, data);

  /* Destroy builder, since we don't need it anymore */
  g_object_unref(G_OBJECT(builder));

  /* Show window. All other widgets are automatically shown by GtkBuilder */
  gtk_widget_show(data->main_window);

  /* Start main loop */
  std::cout << "starting main loop" << std::endl;
  gtk_main();

  return (0);
}
