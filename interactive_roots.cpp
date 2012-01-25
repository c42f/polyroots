#include "interactive_roots.h"

#include <complex>
#include <algorithm>
#include <float.h>
#include <iostream>
#include <math.h>

#include <QtGui/QApplication>
#include <QtGui/QPainter>
#include <QtGui/QResizeEvent>
#include <QtCore/QDebug>

// Deduced from matplotlib's gist_heat colormap
static void colormap(double x, float& r, float& g, float& b)
{
    x = std::min(std::max(0.0, x), 1.0); // clamp
    r = std::min(1.0, x/0.7);
    g = std::max(0.0, (x - 0.477)/(1 - 0.477));
    b = std::max(0.0, (x - 0.75)/(1 - 0.75));
}

typedef std::complex<double> complex;

template<typename T>
inline T abs2(std::complex<T> z)
{
    T x = real(z), y = imag(z);
    return x*x + y*y;
}

double evaluate(complex poly, complex C2, double* bounds, complex* zpows, int n)
{
    using std::min;
//    if(n < 0)
//        return abs2(poly);
    // Unrolled a few times for speed.
    if(n < 3)
    {
        return min(min(min(abs2(poly +    zpows[0] +    zpows[1] +    zpows[2]),
                           abs2(poly +    zpows[0] +    zpows[1] + C2*zpows[2])),
                       min(abs2(poly +    zpows[0] + C2*zpows[1] +    zpows[2]),
                           abs2(poly +    zpows[0] + C2*zpows[1] + C2*zpows[2]))),
                   min(min(abs2(poly + C2*zpows[0] +    zpows[1] +    zpows[2]),
                           abs2(poly + C2*zpows[0] +    zpows[1] + C2*zpows[2])),
                       min(abs2(poly + C2*zpows[0] + C2*zpows[1] +    zpows[2]),
                           abs2(poly + C2*zpows[0] + C2*zpows[1] + C2*zpows[2]))));
    }
    else
    {
        // Check bound on partial sum of current terms and prune if there can
        // be no roots in this region.
        //
        // The fudge factor determines how closely to a root we can cut off
        // the search.  A value of 1 can be used, but a larger value gives
        // better looking results.
        const double fudge = 1;
        if(abs2(poly) > fudge*(*bounds))
            return FLT_MAX;
        return min(evaluate(poly + *zpows,     C2, bounds+1, zpows+1, n-1),
                   evaluate(poly + C2* *zpows, C2, bounds+1, zpows+1, n-1));
    }
}


void minPolys(float* result, int N, int M,
              double x0, double x1, double y0, double y1,
              int degree, complex C2)
{
    int linesdone = 0;
#   pragma omp parallel for schedule(dynamic, 1)
    for(int j = 0; j < M; ++j)
    {
        for(int i = 0; i < N; i+=1)
        {
            // Calculate powers of z for current point
            double x = x0 + (x1-x0) * (i + 0.5)/N;
            double y = y0 + (y1-y0) * (j + 0.5)/M;
            complex z(x,y);
            // Remap z using symmetry to improve bounding performance.
            if(abs(z) > 1)
                z = 1.0/z;
            complex zpowers[degree+1];
            for(int k = 0; k <= degree; ++k)
                zpowers[k] = pow(z, k);
            // Precompute bounds on the absolute value of partial sum of last
            // k terms, via the triangle inequality.
            double bounds[degree+1];
            bounds[degree] = abs(zpowers[degree]);
            for(int k = degree-1; k >= 0; --k)
                bounds[k] = bounds[k+1] + abs(zpowers[k]);
            for(int k = 0; k <= degree; ++k)
                bounds[k] *= bounds[k];
            // Find minimum absolute value of all terms; there are 2^(degree+1)
            // polynomials, but we reduce that by a factor of two by making use
            // of symmetry [for every P(z) there is a -P(z) in the set]
            double minpoly = evaluate(0.0, C2, bounds, zpowers, degree);
            // Weight of 1/|z|^degree implies z <--> 1/z symmetry
            result[N*j + i] = sqrt(minpoly / pow(abs(z),degree));
        }
#       pragma omp critical
        {
            ++linesdone;
            std::cout << 100.0*linesdone/M << "%   \r" << std::flush;
        }
    }
}

//------------------------------------------------------------------------------
RootViewWidget::RootViewWidget()
    : m_coeff2(-1,0),
    m_coeffChangeMode(false),
    m_degree(10),
    m_bbox(-2,-2,4,4)
{ }


void RootViewWidget::keyPressEvent(QKeyEvent* event)
{
    if(event->key() == Qt::Key_Plus)
        ++m_degree;
    else if(event->key() == Qt::Key_Minus)
        --m_degree;
    else if(event->key() == Qt::Key_C)
        m_coeffChangeMode = !m_coeffChangeMode;
    else
    {
        event->ignore();
        return;
    }
    renderImage();
}


void RootViewWidget::resizeEvent(QResizeEvent* event)
{
    // adjust bounding box
    QPointF c = m_bbox.center();
    qreal h = m_bbox.height();
    qreal w = h * event->size().width() / event->size().height();
    m_bbox = QRectF(c.x() - w/2, c.y() - h/2, w, h);
    // re-render
    renderImage();
}


void RootViewWidget::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);
    painter.drawImage(QPoint(0, 0), m_image);
    if(m_coeffChangeMode)
    {
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(Qt::white);
        qreal R = std::min(width(), height())/2;
        QPointF center(width()/2, height()/2);
        painter.drawEllipse(center, R, R);
        painter.setBrush(QBrush(Qt::white, Qt::SolidPattern));
        painter.drawEllipse(center + R*m_coeff2, 4, 4);
    }
}


void RootViewWidget::mousePressEvent(QMouseEvent* event)
{
    m_lastPos = event->pos();
}


void RootViewWidget::mouseMoveEvent(QMouseEvent* event)
{
    if(m_coeffChangeMode)
    {
        // Change polynomial coefficient
        qreal R = std::min(width(), height())/2.0;
        m_coeff2 = 1/R * (event->pos() - QPointF(width()/2.0, height()/2.0));
    }
    else
    {
        QPoint delta = event->pos() - m_lastPos;
        m_lastPos = event->pos();
        if(event->buttons() & Qt::RightButton)
        {
            // zoom bbox
            QPointF c = m_bbox.center();
            qreal scale = exp(2.0 * delta.y()/height());
            qreal w = m_bbox.width() * scale;
            qreal h = m_bbox.height() * scale;
            m_bbox = QRectF(c.x() - w/2, c.y() - h/2, w, h);
        }
        else
        {
            // pan bbox
            m_bbox.translate(-qreal(delta.x())/width()*m_bbox.width(),
                            -qreal(delta.y())/height()*m_bbox.height());
        }
    }
    renderImage();
}


void RootViewWidget::renderImage()
{
    std::cout
        << "degree = " << m_degree
        << ", bbox = [" << m_bbox.x() << ", " << m_bbox.x() + m_bbox.width()
        << "] x [" << m_bbox.y() << ", " << m_bbox.y() + m_bbox.height()
        << "], C2 = " << m_coeff2.x() << " + " << m_coeff2.y() << "i\n";
    int N = size().width();
    int M = size().height();
    m_image = QImage(N, M, QImage::Format_RGB32);
    float* minP = new float[N*M];
    // Compute second coefficient.  (Assumption: always scale larger
    // coefficient to equal 1)
    complex C2(m_coeff2.x(), m_coeff2.y());
    if(abs(C2) > 1)
        C2 = 1.0/C2;
    minPolys(minP, N, M, m_bbox.x(), m_bbox.x()+m_bbox.width(),
             m_bbox.y(), m_bbox.y() + m_bbox.height(), m_degree, C2);
    for(int j = 0; j < M; ++j)
    {
        QRgb* pix = reinterpret_cast<QRgb*>(m_image.scanLine(j));
        for(int i = 0; i < N; ++i)
        {
            float r,g,b;
            colormap(1 - (pow(minP[j*N + i], 0.01) - 0.95)/0.1, r,g,b);
            pix[i] = qRgb(lround(r*255), lround(g*255), lround(b*255));
        }
    }
    delete[] minP;
    update();
}


int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    RootViewWidget view;
    view.show();

    return app.exec();
}

