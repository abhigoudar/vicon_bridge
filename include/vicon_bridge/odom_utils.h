#pragma once

#include <Eigen/Dense>

constexpr double epsilon = 1e-10;

struct Pose
{
    Pose()
    {
        stamp = -1;
        pos.fill(0);
        ori.setIdentity();
    }
    //
    ~Pose()
    {

    }
    //
    void reset()
    {
        stamp = -1;
        pos.fill(0);
        ori.setIdentity();
    }
    //
    double stamp;
    Eigen::Vector3d pos;
    Eigen::Quaterniond ori;
};

const Eigen::Vector3d quaternionLog(const Eigen::Quaterniond& q)
{
    using std::abs;
    using std::atan2;
    using std::sqrt;
    double squared_n = q.vec().squaredNorm();
    double w = q.w();

    double two_atan_nbyw_by_n;
    double theta;

    if (squared_n < epsilon *epsilon) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      assert(abs(w) >= epsilon && "Quaternion ({}) should be normalized!");
      double squared_w = w * w;
      two_atan_nbyw_by_n = 2.0 / w - (2.0 / 3.0) * (squared_n) / (w * squared_w);
      theta = 2 * squared_n / w;
    } 
    else 
    {
      double n = sqrt(squared_n);
      double atan_nbyw = (w < 0) ? atan2(-n, -w) : atan2(n, w);
      two_atan_nbyw_by_n = 2 * atan_nbyw / n;
      theta = two_atan_nbyw_by_n * n;
    }

    return (two_atan_nbyw_by_n * q.vec());
}

// Simple moving average filter
template<typename T>
class MovingWindowFilter
{
public:
    MovingWindowFilter(int wndSize);

    void reset();

    const T mean() const;

    void push(const T& datum);

private:

    T sum;
    int size;
    std::deque<T> window;
};

template<>
MovingWindowFilter<double>::MovingWindowFilter(int wndSize)
{
  sum = 0;
  size = wndSize;
  window.clear();
}

template<>
void MovingWindowFilter<double>::reset()
{
  sum = 0;
  window.clear();
}

template<>
const double MovingWindowFilter<double>::mean() const
{
  return (window.size() > 0 ? (1.0/(int)window.size()) * sum : 0.0);
}

template<>
MovingWindowFilter<Eigen::Vector3d>::MovingWindowFilter(int wndSize)
{
  sum.fill(0);
  size = wndSize;
  window.clear();
}

template<>
void MovingWindowFilter<double>::push(const double& datum)
{
  if((int)window.size() < size)
  {
    sum += datum;
  }
  else
  {
    sum = (sum - window.front()) + datum;
    window.pop_front();
  }

  window.push_back(datum);
}

template<>
void MovingWindowFilter<Eigen::Vector3d>::reset()
{
  sum.fill(0);
  window.clear();
}

template<>
const Eigen::Vector3d MovingWindowFilter<Eigen::Vector3d>::mean() const
{
  return (window.size() > 0 ? (1.0/(int)window.size()) * sum : 
    Eigen::Vector3d(0., 0., 0.));
}

template<>
void MovingWindowFilter<Eigen::Vector3d>::push(const Eigen::Vector3d& datum)
{
  if((int)window.size() < size)
  {
    sum += datum;
  }
  else
  {
    sum = (sum - window.front()) + datum;
    window.pop_front();
  }

  window.push_back(datum);
}