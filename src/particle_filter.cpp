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
  const char* delim = "";
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

Point LocalToGlobal(Point map_pt, Particle p) {
  double x = map_pt.x * cos(p.theta) - map_pt.y * sin(p.theta) + p.x;
  double y = map_pt.x * sin(p.theta) + map_pt.y * cos(p.theta) + p.y;
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

  const size_t kTotalParticles = 100;

  std::default_random_engine gen;
  std::normal_distribution<double> x_dist(x, std[0]), y_dist(y, std[1]),
      theta_dist(theta, std[2]);

  for (int i = 0; i < kTotalParticles; ++i) {
    const double sx = x_dist(gen);
    const double sy = y_dist(gen);
    const double stheta = theta_dist(gen);

    particles.push_back(Particle{i, sx, sy, stheta, 1.0, {}, {}, {}});
  }

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
    const double theta_new = particle.theta + delta_t * yaw_rate;

    double x_new, y_new;
    if (fabs(yaw_rate) > 0.00001) {
      const double k = velocity / yaw_rate;
      x_new = particle.x + k * (sin(theta_new) - sin(particle.theta));
      y_new = particle.y + k * (cos(particle.theta) - cos(theta_new));
    } else {
      x_new = particle.x + velocity * delta_t * cos(particle.theta);
      y_new = particle.y + velocity * delta_t * sin(particle.theta);
    }

    const double x_noise = x_dist(gen);
    const double y_noise = y_dist(gen);
    const double theta_noise = theta_dist(gen);

    particle.x = x_new + x_noise;
    particle.y = y_new + y_noise;
    particle.theta = theta_new + theta_noise;
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
    std::vector<Map::single_landmark_s> possible_landmarks;
    for (auto& lmark : map_landmarks.landmark_list) {
      const double distance =
          dist(lmark.x_f, lmark.y_f, particle.x, particle.y);
      if (distance <= sensor_range) {
        possible_landmarks.push_back(lmark);
      }
    }

    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    std::vector<double> lmark_x;
    std::vector<double> lmark_y;

    for (const auto& obs : observations) {
      const Point obs_on_map = LocalToGlobal(Point{obs.x, obs.y}, particle);

      double best_distance = std::numeric_limits<double>::max();
      bool found = false;
      size_t best_idx = 0;
      for (size_t i = 0; i < possible_landmarks.size(); ++i) {
        double d = dist(obs_on_map.x, obs_on_map.y, possible_landmarks[i].x_f,
                        possible_landmarks[i].y_f);
        if (!found || (d < best_distance)) {
          found = true;
          best_distance = d;
          best_idx = i;
        }
      }

      if (found) {
        const auto& lmark = possible_landmarks[best_idx];
        associations.push_back(lmark.id_i);
        sense_x.push_back(obs_on_map.x);
        sense_y.push_back(obs_on_map.y);
        lmark_x.push_back(lmark.x_f);
        lmark_y.push_back(lmark.y_f);
        possible_landmarks.erase(possible_landmarks.begin() + best_idx);
      }
    }

    double weight = 1.0;

    for (size_t i = 0; i < associations.size(); ++i) {
      const double x_delta = lmark_x[i] - sense_x[i];
      const double y_delta = lmark_y[i] - sense_y[i];

      const double kx =
          (x_delta * x_delta) / (std_landmark[0] * std_landmark[0]);
      const double ky =
          (y_delta * y_delta) / (std_landmark[1] * std_landmark[1]);
      const double pdf = exp(-0.5 * (kx + ky)) /
                         (2 * M_PI * std_landmark[0] * std_landmark[1]);
      weight *= pdf;
    }

    particle.weight = weight;
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<double> weights;
  for (const auto& p : particles) {
    weights.push_back(p.weight);
  }

  std::discrete_distribution<> dd(begin(weights), end(weights));
  std::default_random_engine gen;

  std::vector<Particle> new_particles;
  for (size_t i = 0; i < particles.size(); ++i) {
    new_particles.push_back(particles[dd(gen)]);
  }
  particles = new_particles;
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
