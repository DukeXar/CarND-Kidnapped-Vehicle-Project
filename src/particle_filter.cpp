/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

namespace {
template <typename T>
std::string VecToS(const std::vector<T>& v) {
  std::ostringstream oss;
  bool delim = "";
  for (const auto& i : v) {
    oss << delim << i;
    delim = " ";
  }
  return oss.str();
}

struct Point {
  double x;
  double y;
};

Point ConvertGlobalToLocal(Point map_pt, Point p_pt, double p_theta) {
  // Translate from global to local
  double tx = map_pt.x - p_pt.x;
  double ty = map_pt.y - p_pt.y;

  // Then rotate CCW
  double x = tx * cos(p_theta) - ty * sin(p_theta);
  double y = tx * sin(p_theta) + ty * cos(p_theta);
  return Point{x, y};
}

}  // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).
  std::default_random_engine gen;
  std::normal_distribution<double> x_dist(x, std[0]), y_dist(y, std[1]),
      theta_dist(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sx = x_dist(gen);
    double sy = y_dist(gen);
    double stheta = theta_dist(gen);

    particles.push_back(Particle{i, sx, sy, stheta, 1.0, {}, {}, {}});
  }

  weights.assign(num_particles, 1.0);

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  std::default_random_engine gen;
  std::normal_distribution<double> x_dist(0, std_pos[0]), y_dist(0, std_pos[1]),
      theta_dist(0, std_pos[2]);

  for (auto& particle : particles) {
    double theta_new = particle.theta + delta_t * yaw_rate;
    double k = velocity / yaw_rate;
    double x_new = particle.x + k * (sin(theta_new) - sin(particle.theta));
    double y_new = particle.y + k * (cos(particle.theta) - cos(theta_new));

    double x_noise = x_dist(gen);
    double y_noise = y_dist(gen);
    double theta_noise = theta_dist(gen);

    particle.x = x_new + x_noise;
    particle.y = y_new + y_noise;
    particle.theta = std::fmod(theta_new + theta_noise, 2 * M_PI);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems.
  //   Keep in mind that this transformation requires both rotation AND
  //   translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html)

  for (auto& particle : particles) {
    std::vector<LandmarkObs> predicted_observations;

    for (auto& lmark : map_landmarks.landmark_list) {
      const double distance =
          dist(lmark.x_f, lmark.y_f, particle.x, particle.y);

      if (distance <= sensor_range) {
        const Point pt =
            ConvertGlobalToLocal(Point(lmark.x_f, lmark.y_f),
                                 Point(particle.x, particle.y), particle.theta);
        predicted_observations.push_back(LandmarkObs{lmark.id, pt.x, pt.y});
      }
    }

    std::vector<LandmarkObs> associated_observations;
    std::vector<LandmarkObs> associated_predictions;

    for (const auto& actual : observations) {
      double min_so_far = std::numeric_limits<double>::max();
      size_t found_idx = 0;
      bool found = false;
      for (size_t i = 0; i < predicted_observations.size(); ++i) {
        const auto& candiate = predicted_observations[i];
        double distance = dist(candidate.x, candidate.y, actual.x, actual.y);
        if (distance < min_so_far) {
          found_idx = i;
          found = true;
          min_so_far = distance;
        }
      }

      // TODO(dukexar): Could have several landmarks associated with same
      // observation? Seems wrong.

      if (found) {
        associated_observations.push_back(LandmarkObs{
            predicted_observations[found_idx].id, actual.x, actual.y});
        associated_predictions.push_back(predicted_observations[found_idx]);
      }
    }

    double weight = 1;
    for (size_t i = 0; i < associated.size(); ++i) {
      double x_delta =
          associated_observations[i].x - associated_predictions[i].x;
      double y_delta =
          associated_observations[i].y - associated_predictions[i].y;
    }

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    for (const auto& assoc : associated) {
      particle.associations.push_back(assoc.id);
      particle.sense_x.push_back(assoc.x);
      particle.sense_y.push_back(assoc.y);
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

std::string ParticleFilter::getAssociations(Particle best) {
  return VecToS(best.associations);
}

std::string ParticleFilter::getSenseX(Particle best) {
  return VecToS(best.sense_x);
}

std::string ParticleFilter::getSenseY(Particle best) {
  return VecToS(best.sense_y);
}
