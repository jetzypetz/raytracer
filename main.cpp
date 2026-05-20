// g++ -Wall -Werror -fopenmp -std=c++11 main.cpp

#define _CRT_SECURE_NO_WARNINGS 1
#include <omp.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <limits>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef EPS
#define EPS 0.0000000001
#endif

#ifndef ANTIALIASING
#define ANTIALIASING 10
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

static std::default_random_engine engine[32];
static std::uniform_real_distribution<double> uniform(0, 1);

int counter = 0;

double sqr(double x) { return x * x; };

class Vector {
public:
	explicit Vector(double x = 0, double y = 0, double z = 0) {
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}
	double norm2() const {
		return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
	}
	double norm() const {
		return sqrt(norm2());
	}
	Vector& normalize() {
		double n = norm();
		data[0] /= n;
		data[1] /= n;
		data[2] /= n;
		return *this;
	}

	// operations added by deepseek ai for cleaner code
	Vector& operator+=(const Vector& other) {
        data[0] += other[0];
        data[1] += other[1];
        data[2] += other[2];
        return *this;
    }
    
    Vector& operator-=(const Vector& other) {
        data[0] -= other[0];
        data[1] -= other[1];
        data[2] -= other[2];
        return *this;
    }
    
    Vector& operator*=(double scalar) {
        data[0] *= scalar;
        data[1] *= scalar;
        data[2] *= scalar;
        return *this;
    }
    
    Vector& operator/=(double scalar) {
        data[0] /= scalar;
        data[1] /= scalar;
        data[2] /= scalar;
        return *this;
    }

	double operator[](int i) const { return data[i]; };
	double& operator[](int i) { return data[i]; };
	double data[3];
};

Vector operator+(const Vector& a, const Vector& b) {
	return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
	return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const double a, const Vector& b) {
	return Vector(a*b[0], a*b[1], a*b[2]);
}
Vector operator*(const Vector& a, const double b) {
	return Vector(a[0]*b, a[1]*b, a[2]*b);
}
Vector operator/(const Vector& a, const double b) {
	return Vector(a[0] / b, a[1] / b, a[2] / b);
}
double dot(const Vector& a, const Vector& b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b) {
	return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}
Vector mult(const Vector& a, const Vector& b) {
	return Vector(a[0] * b[0], a[1] * b[1], a[2] * b[2]);
}
int offset(const Vector& a, const Vector& b) {
	return (dot(a, b) < 0) ? 1 : -1;
}

Vector random_cos(const Vector &N) {

	double r1 = uniform(engine[omp_get_thread_num()]);
	double r2 = uniform(engine[omp_get_thread_num()]);

	double x = cos(2 * M_PI * r1) * sqrt(1 - r2);
	double y = sin(2 * M_PI * r1) * sqrt(1 - r2);
	double z = sqrt(r2);
	
	Vector T1;
	Vector T2;

	int smallest = 0;
	if (N[1] * N[1] < N[smallest] * N[smallest]) {smallest = 1;}
	if (N[2] * N[2] < N[smallest] * N[smallest]) {smallest = 2;}
	
	switch (smallest) {
		case 0:
			T1 = Vector(0, -N[2], N[1]);
			break;
		case 1:
			T1 = Vector(N[2], 0, -N[0]);
			break;
		case 2:
			T1 = Vector(-N[1], N[0], 0);
			break;
	}

	T1.normalize();

	T2 = cross(N, T1);

	return (x * T1 + y * T2 + z * N);
}

void boxMuller(double &stdev, double& x, double& y) {
	double r1 = uniform(engine[omp_get_thread_num()]);
	double r2 = uniform(engine[omp_get_thread_num()]);

	x = sqrt(-2 * log (r1)) * cos(-2 * M_PI * r2) * stdev;
	y = sqrt(-2 * log (r1)) * sin(-2 * M_PI * r2) * stdev;
}

class Ray {
public:
	Ray(const Vector& origin, const Vector& unit_direction) : O(origin), u(unit_direction) {};
	Vector O, u;
};

class Object {
public:
	Object(const Vector& albedo, bool mirror = false, bool transparent = false) : albedo(albedo), mirror(mirror), transparent(transparent) {};

	virtual bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const = 0;

	Vector albedo;
	bool mirror, transparent;
};

class Sphere : public Object {
public:
	Sphere(const Vector& center, double radius, const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent), R(radius), C(center) {};

	// returns true iif there is an intersection between the ray and the sphere
	// if there is an intersection, also computes the point of intersection P, 
	// t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
	// and the unit normal N
	bool intersect(const Ray& ray, Vector& P, double &t, Vector& N) const {
		// TODO (lab 1) : compute the intersection (just true/false at the begining of lab 1, then P, t and N as well)
		double Delta = sqr(R) + sqr(dot(ray.u, ray.O - C)) - dot(ray.O - C, ray.O - C);
		
		if (Delta < 0) { return false; }

		t = dot(ray.u, C - ray.O) - sqrt(Delta);
		
		if (t < 0) {
			t = t + 2 * sqrt(Delta);
			if (t < 0) { return false; }
		}
		P = ray.O + t * ray.u;
		N = P - C;
		N.normalize();
		return true;
	}

	double R;
	Vector C;
};

class TriangleMesh : public Object {
public:
	TriangleMesh(const Vector& albedo, bool mirror = false, bool transparent = false) : ::Object(albedo, mirror, transparent) {};
	
	const Vector& A(const size_t& i) const {
		return vertices[triangles[3*i]];
	}
	const Vector& B(const size_t& i) const {
		return vertices[triangles[3*i+1]];
	}
	const Vector& C(const size_t& i) const {
		return vertices[triangles[3*i+2]];
	}

	bool intersect_one(const Ray& ray, Vector& P, double& t, Vector& N, const size_t& i) const {
		Vector e1 = A(i) - B(i);
		Vector e2 = A(i) - C(i);

		N = cross(e1, e2);
		
		t = dot(A(i) - ray.O, N) / dot(ray.u, N);

		if (t <= 0) {
			return false;
		}

		double beta  = dot(e2, cross(A(i) - ray.O, ray.u)) / dot(ray.u, N);
		double gamma = dot(e1, cross(A(i) - ray.O, ray.u)) / dot(ray.u, N);
		double alpha = 1 - beta - gamma;

		if (!((beta < 1) && (beta > 0) && (alpha < 1) && (alpha > 0) && (gamma < 1) && (gamma > 0))) {
			return false;
		}

		P = ray.O + t * ray.u;
		N.normalize();

		return true;
	}

	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N) const {
		t = __DBL_MAX__;

		Vector Cand_P, Cand_N;
		double Cand_t;
		bool found = false;

		for (size_t i=0; i<total_triangles; i++) {
			if (intersect_one(ray, Cand_P, Cand_t, Cand_N, i) && Cand_t < t) {
				found = true;
				N = Cand_N;
				P = Cand_P;
				t = Cand_t;
			}
		}
		
		return found;
	}

	size_t total_triangles;
	std::vector<Vector> vertices;
	std::vector<size_t> triangles;
};


class Scene {
public:
	Scene() {};
	void addObject(const Object* obj) {
		objects.push_back(obj);
	}

	// returns true iif there is an intersection between the ray and any object in the scene
    // if there is an intersection, also computes the point of the *nearest* intersection P, 
    // t>=0 the distance between the ray origin and P (i.e., the parameter along the ray)
    // and the unit normal N. 
	// Also returns the index of the object within the std::vector objects in object_id
	bool intersect(const Ray& ray, Vector& P, double& t, Vector& N, int &object_id) const  {

		// TODO (lab 1): iterate through the objects and check the intersections with all of them, 
		// and keep the closest intersection, i.e., the one if smallest positive value of t
		
		t = __DBL_MAX__;

		Vector Cand_P, Cand_N; // Candidate values
		double Cand_t;
		bool found = false;

		for (unsigned long int i = 0; i < objects.size(); i++) {
			if ((objects[i]->intersect(ray, Cand_P, Cand_t, Cand_N)) && Cand_t < t) {
				found = true;
				t = Cand_t;
				N = Cand_N;
				P = Cand_P;
				object_id = i;
			}
		}

		return found;
	}


	// return the radiance (color) along ray
	Vector getColor(const Ray& ray, int recursion_depth) {

		if (recursion_depth >= max_light_bounce) return Vector(0, 0, 0);

		// TODO (lab 1) : if intersect with ray, use the returned information to compute the color ; otherwise black 
		// in lab 1, the color only includes direct lighting with shadows

		Vector P, N;
		double t;
		int object_id;

		Vector return_color(0, 0, 0);

		if (intersect(ray, P, t, N, object_id)) {

			if (objects[object_id]->mirror) {

				// return getColor in the reflected direction, with recursion_depth+1 (recursively)

				return getColor(Ray(P + EPS * N * offset(ray.u, N), ray.u - (2 * dot(ray.u, N) * N)), recursion_depth+1);
			} else {
				// test if there is a shadow by sending a new ray
				// if shadow, do nothing
				// if no shadow, return colour set to albedo

				Vector P_prime, N_prime;
				double t_prime;
				int oid_prime; // unused, just needed for intersect

				Vector light_direction = light_position - P;
				light_direction.normalize();

				if (!intersect(Ray(P + EPS * N * offset(ray.u, N), light_direction), P_prime, t_prime, N_prime, oid_prime) // no obstruction with light
				|| t_prime > sqrt(dot(light_position - P, light_position - P))) { // or obstuctor dist is further than light pos
					
					Vector S_P = light_position - P;
					double d = sqrt(dot(S_P, S_P));
					
					S_P.normalize();
					
					return_color = (objects[object_id]->albedo * light_intensity * dot(N, S_P)) / (4 * M_PI * M_PI * d * d);
				}

				Ray random_ray(P + EPS * N * offset(ray.u, N), random_cos(N));

				Vector indirect_light = mult(getColor(random_ray, recursion_depth + 1), objects[object_id]->albedo);

				return_color += indirect_light;

				return return_color;
			}

		} else {
			return Vector(0, 0, 0);
		}
	}

	std::vector<unsigned char> render(const int W = 512, const int H = 512) {

		std::vector<unsigned char> image(W * H * 3, 0);
		
		#pragma omp parallel for schedule(dynamic, 1)

		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {

				// TODO (lab 1) : correct ray_direction so that it goes through each pixel (j, i)			
				
				// TODO (lab 2) : add Monte Carlo / averaging of random ray contributions here
				// TODO (lab 2) : add antialiasing by altering the ray_direction here
				// TODO (lab 2) : add depth of field effect by altering the ray origin (and direction) here

				Vector color(0., 0., 0.);
				
				for (int k=0; k<ANTIALIASING; k++) {
					double random_x, random_y;
					double stdev = 0.5;
				
					boxMuller(stdev, random_x, random_y);

					if ((random_x * random_x < 0.25) && (random_y * random_y < 0.25)) {
						Vector ray_direction = Vector(j - W/2 + 0.5 + random_x, H/2 - i - 0.5 + random_y, -W / (2 * tan(fov / 2)));
						ray_direction.normalize();
		
						Ray ray = Ray(camera_center, ray_direction);
	
						color += getColor(ray, 0);
					}

				}

				color *= 255. / ANTIALIASING;

				image[(i * W + j) * 3 + 0] = std::min(255., std::max(0., 255. * std::pow(color[0] / 255., 1. / gamma)));
				image[(i * W + j) * 3 + 1] = std::min(255., std::max(0., 255. * std::pow(color[1] / 255., 1. / gamma)));
				image[(i * W + j) * 3 + 2] = std::min(255., std::max(0., 255. * std::pow(color[2] / 255., 1. / gamma)));
			}
		}

		return image;
	}	

	std::vector<const Object*> objects;

	Vector camera_center, light_position;
	double fov, gamma, light_intensity;
	int max_light_bounce;
};

int main() {
	TriangleMesh triangle(Vector(0.8, 0.2, 0.5), false, false);
	
	triangle.vertices.push_back(Vector(0, 50, -320));
	triangle.vertices.push_back(Vector(-100, 20, -220));
	triangle.vertices.push_back(Vector(-20, -30, -220));
	
	triangle.triangles.push_back(0);
	triangle.triangles.push_back(1);
	triangle.triangles.push_back(2);
	
	triangle.total_triangles = 1;
	
	Sphere center_sphere(Vector(-100, -100, -400), 50., Vector(0.8, 0.8, 0.8));
	Sphere off_center_sphere(Vector(100, -100, -420), 40., Vector(0.8, 0.8, 0.8), true);
	Sphere far_sphere(Vector(200, 100, -4800), 3000, Vector(1, 0.5, 0.7));
	Sphere wall_left(Vector(-9800, 300, -300), 9500, Vector(0.5, 0.8, 0.1));
	Sphere wall_right(Vector(9800, 300, -300), 9500, Vector(0.9, 0.2, 0.3));
	Sphere wall_front(Vector(0, 300, -10100), 9500, Vector(0.1, 0.6, 0.7));
	Sphere wall_behind(Vector(0, 300, 9500), 9000, Vector(0.8, 0.2, 0.9));
	Sphere ceiling(Vector(0, 9800, -300), 9500, Vector(0.3, 0.5, 0.3));
	Sphere floor(Vector(0, -9700, -300), 9500, Vector(0.6, 0.5, 0.7));
	
	for (int i = 0; i<32; i++) {
		engine[i].seed(i);
	}
	
	Scene scene;
	scene.camera_center = Vector(0, 0, 55);
	scene.light_position = Vector(-100, 200, -200);
	scene.light_intensity = 1E7;
	scene.fov = 60 * M_PI / 180.;
	scene.gamma = 2.2;
	scene.max_light_bounce = 2;
	
	scene.addObject(&triangle);
	scene.addObject(&center_sphere);
	scene.addObject(&off_center_sphere);
	scene.addObject(&wall_left);
	scene.addObject(&wall_right);
	scene.addObject(&wall_front);
	scene.addObject(&wall_behind);
	scene.addObject(&ceiling);
	scene.addObject(&floor);
	
	int W = 512;
	int H = 512;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	std::vector<unsigned char> image = scene.render(W, H);
	
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	
	stbi_write_png("image.png", W, H, 3, &image[0], 0);

	return 0;
}
